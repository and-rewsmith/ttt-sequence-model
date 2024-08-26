import math

from torch import nn, Tensor
import torch
from torch.optim import SGD
from typing import Callable, Self
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot  # type: ignore
from torch.nn import functional as F
import wandb

# data
NUM_SAMPLES = 1000
VOCAB_SIZE = 15
SEQUENCE_LEN = 40

# training
EPOCHS = 10
BATCH_SIZE = 50

TTT_BASE_INNER_LEARNING_RATE = 1e-4
TTT_INNER_LEARNING_RATE_LEARNING_RATE = 1e-1
TTT_OUTER_LEARNING_RATE = 1e-3
EMBEDDING_DIM = 22
LOW_PASS_FILTER_DIM = 10
MINIBATCH_SIZE = 5
DROPOUT = 0.0


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        out: torch.Tensor = self.dropout(x)
        return out


class SequentialNumbers(Dataset):
    def __init__(self: Self, num_samples: int, sequence_len: int) -> None:
        self.num_samples = num_samples
        self.sequence_len = sequence_len

    def __len__(self: Self) -> int:
        return self.num_samples

    def __getitem__(self: Self, idx: int) -> Tensor:
        sequence = torch.zeros(self.sequence_len, VOCAB_SIZE)

        for i in range(self.sequence_len):
            hot_index = (idx + i) % VOCAB_SIZE
            sequence[i, hot_index] = 1.0

        return sequence


class TTTInner(nn.Module):
    def __init__(self: Self, mini_batch_size: int, filter_dim: int,
                 get_theta_k: Callable[[], torch.nn.Parameter],
                 get_theta_q: Callable[[], torch.nn.Parameter],
                 get_theta_v: Callable[[], torch.nn.Parameter],
                 get_inner_learning_rate: Callable[[Tensor], Tensor]) -> None:
        super(TTTInner, self).__init__()

        self.mini_batch_size = mini_batch_size
        self.filter_dim = filter_dim
        self.w = nn.Linear(filter_dim, filter_dim)
        torch.nn.init.kaiming_uniform_(self.w.weight)

        self.get_theta_k = get_theta_k
        self.get_theta_q = get_theta_q
        self.get_theta_v = get_theta_v
        self.get_inner_learning_rate = get_inner_learning_rate

    def online_inference(self: Self, src: torch.Tensor) -> torch.Tensor:
        _sequences, _batches, _features = src.shape
        src = torch.split(src, self.mini_batch_size)  # type: ignore[assignment]

        outputs = []
        total_loss = 0
        for minibatch in src:
            minibatch_seq, minibatch_batch, minibatch_features = minibatch.shape

            # compute views
            train_view = minibatch @ self.get_theta_k()  # type: ignore[operator]
            label_view = minibatch @ self.get_theta_v()  # type: ignore[operator]
            test_view = minibatch @ self.get_theta_q()  # type: ignore[operator]

            assert train_view.shape == (minibatch_seq, minibatch_batch, self.filter_dim)
            assert label_view.shape == (minibatch_seq, minibatch_batch, self.filter_dim)

            # reconstruction loss
            reconstruction_target = label_view - train_view  # type: ignore[operator]
            w_train_view = self.w(train_view)
            loss = nn.MSELoss()(w_train_view, reconstruction_target)
            total_loss += loss

            # compute gradients for `w` and manually update
            gradients = grad(loss, list(self.w.parameters()), create_graph=True)
            assert gradients[0].shape == self.w.weight.shape

            wandb.log({"w_grad": gradients[0].norm()})
            wandb.log({"w_bias_grad": gradients[1].norm()})

            # calculate the learned inner learning rate for each parameter and shape appropriately
            inner_learning_rate = self.get_inner_learning_rate(minibatch)
            inner_learning_rate = inner_learning_rate.reshape(-1, self.filter_dim ** 2)
            inner_learning_rate = inner_learning_rate.mean(dim=0)
            inner_learning_rate = inner_learning_rate.reshape(self.filter_dim, self.filter_dim)
            inner_learning_rate_bias = inner_learning_rate.mean(dim=1)

            # TODO: consider adding layer norm here to stabilize batch effects of averaging

            wandb.log({"inner_learning_rate": inner_learning_rate.norm()})
            wandb.log({"inner_learning_rate_bias": inner_learning_rate_bias.norm()})
            wandb.log(
                {"inner_learning_rate_specific_index": inner_learning_rate[0][0]})

            updated_weight = self.w.weight - inner_learning_rate * gradients[0]
            updated_bias = self.w.bias - inner_learning_rate_bias * gradients[1]

            # calculate output using updated `w_bar`
            z = torch.nn.functional.linear(test_view, updated_weight, updated_bias) + test_view
            outputs.append(z)

            # this was intended to stop grads from flowing back to weights (minor optimization)
            # doesn't seem to work though
            # leaving it
            self.w.weight.requires_grad_(False)
            self.w.bias.requires_grad_(False)

            # update `w` with `w_bar` which resets computation graph
            with torch.no_grad():
                self.w.weight = nn.Parameter(updated_weight, requires_grad=True)
                self.w.bias = nn.Parameter(updated_bias, requires_grad=True)

        average_loss = total_loss / len(src)
        wandb.log({"inner_loss": average_loss})

        return torch.concat(outputs, dim=0)


class TTTHead(nn.Module):
    def __init__(self: Self, mini_batch_size: int, input_dim: int, filter_dim: int,
                 ttt_base_inner_learning_rate: float) -> None:
        super(TTTHead, self).__init__()

        self.ttt_base_inner_learning_rate = ttt_base_inner_learning_rate

        self.theta_k = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_v = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_q = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_o = nn.Parameter(torch.randn(filter_dim, input_dim))
        self.inner_learning_rate_params = nn.Linear(input_dim, filter_dim ** 2)

        self.inner = TTTInner(mini_batch_size, filter_dim=filter_dim,
                              get_theta_k=self.get_theta_k,
                              get_theta_q=self.get_theta_q, get_theta_v=self.get_theta_v,
                              get_inner_learning_rate=self.get_inner_learning_rate)

        torch.nn.init.kaiming_uniform_(self.theta_k)
        torch.nn.init.kaiming_uniform_(self.theta_v)
        torch.nn.init.kaiming_uniform_(self.theta_q)
        torch.nn.init.kaiming_uniform_(self.theta_o)
        torch.nn.init.kaiming_uniform_(self.inner_learning_rate_params.weight)

        self.mini_batch_size = mini_batch_size
        self.input_dim = input_dim
        self.low_pass_filter_dim = filter_dim

    def train_head(self: Self, input: torch.Tensor) -> torch.Tensor:
        sequences, batches, features = input.shape

        outputs = self.inner.online_inference(input)
        assert outputs.shape == (sequences, batches, self.low_pass_filter_dim)

        outputs: Tensor = outputs @ self.theta_o  # type: ignore
        assert outputs.shape == (sequences, batches, self.input_dim)
        return outputs

    def get_theta_k(self: Self) -> torch.nn.Parameter:
        return self.theta_k

    def get_theta_q(self: Self) -> torch.nn.Parameter:
        return self.theta_q

    def get_theta_v(self: Self) -> torch.nn.Parameter:
        return self.theta_v

    def get_inner_learning_rate(self: Self, input: torch.Tensor) -> Tensor:
        pre_sigmoid = self.inner_learning_rate_params(input)
        wandb.log({"pre_sigmoid": pre_sigmoid.mean()})
        post_sigmoid = self.ttt_base_inner_learning_rate * F.sigmoid(pre_sigmoid)
        wandb.log({"post_sigmoid": post_sigmoid.mean()})
        return post_sigmoid


class TTTBlock(nn.Module):

    # TODO: add residual connection
    # TODO: add layer norm at beginning and end of block
    # TODO: recursive structure
    def __init__(self: Self, mini_batch_size: int, embedding_dim: int, filter_dim: int,
                 ttt_base_inner_learning_rate: float) -> None:
        super(TTTBlock, self).__init__()

        self.ttt_head = TTTHead(mini_batch_size=mini_batch_size, input_dim=embedding_dim,
                                filter_dim=filter_dim,
                                ttt_base_inner_learning_rate=ttt_base_inner_learning_rate)

    def train_block(self, input: torch.Tensor) -> torch.Tensor:
        sequences, batches, features = input.shape
        outputs = self.ttt_head.train_head(input)
        return outputs

    def set_grad_fn(self, alter_grad_fn: Callable[[bool], None]) -> None:
        self.ttt_head.set_grad_fn(alter_grad_fn)

    def set_zero_grad_fn(self, zero_grad_fn: Callable[[], None]) -> None:
        self.ttt_head.set_zero_grad_fn(zero_grad_fn)


class TTTModel(nn.Module):

    def __init__(
            self: Self, mini_batch_size: int, embedding_dim: int, filter_dim: int, ttt_outer_learning_rate: float,
            ttt_base_inner_learning_rate: float, ttt_inner_learning_rate_learning_rate: float, vocab_size: int,
            num_layers: int, dropout: float) -> None:
        super(TTTModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)

        self.ttt_blocks = nn.ModuleList([TTTBlock(mini_batch_size, embedding_dim, filter_dim,
                                                  ttt_base_inner_learning_rate)
                                        for _ in range(num_layers)])

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        params = []
        for block in self.ttt_blocks:
            params.extend([block.ttt_head.theta_k, block.ttt_head.theta_q,
                           block.ttt_head.theta_v, block.ttt_head.theta_o])
        params.extend(self.pos_encoder.parameters())
        params.extend(self.encoder.parameters())
        params.extend(self.lm_head.parameters())

        self.optim = SGD(params, lr=ttt_outer_learning_rate)

        params = []
        for block in self.ttt_blocks:
            params.extend(block.ttt_head.inner_learning_rate_params.parameters())
        self.optim_inner_lr = SGD(
            params,
            lr=ttt_inner_learning_rate_learning_rate)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self: Self, src: torch.Tensor) -> torch.Tensor:
        assert src.shape == (SEQUENCE_LEN, BATCH_SIZE)
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)

        assert src.shape == (SEQUENCE_LEN, BATCH_SIZE, EMBEDDING_DIM)

        output = src
        for block in self.ttt_blocks:
            output = block.train_block(output)

        output: Tensor = self.lm_head(output)  # type: ignore
        assert output.shape == (SEQUENCE_LEN, BATCH_SIZE, self.vocab_size)

        return output

    def train_model(self: Self, src: torch.Tensor) -> torch.Tensor:
        self.optim.zero_grad()
        self.optim_inner_lr.zero_grad()

        assert src.shape == (SEQUENCE_LEN, BATCH_SIZE)
        shifted_labels = src[1:, :]

        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)

        assert src.shape == (SEQUENCE_LEN, BATCH_SIZE, EMBEDDING_DIM)

        output = src
        for block in self.ttt_blocks:
            output = block.train_block(output)

        # trim the last predicted token off of the output (cannot train on it)
        output = output[:-1, :, :]

        # lm head
        output = self.lm_head(output)
        assert output.shape == (SEQUENCE_LEN - 1, BATCH_SIZE, self.vocab_size)

        # loss reshaping
        output = output.reshape(-1, self.vocab_size)
        assert output.shape == ((SEQUENCE_LEN - 1) * BATCH_SIZE, self.vocab_size)
        shifted_labels = shifted_labels.reshape(-1)
        assert shifted_labels.shape == ((SEQUENCE_LEN - 1) * BATCH_SIZE,)

        loss = self.criterion(output, shifted_labels)
        wandb.log({"outer_loss": loss.item()})

        assert list(self.encoder.parameters())[0].grad is None
        for block in self.ttt_blocks:
            assert block.ttt_head.inner.w.weight.grad is None
            assert block.ttt_head.theta_k.grad is None
            assert block.ttt_head.theta_q.grad is None
            assert block.ttt_head.theta_v.grad is None
            assert block.ttt_head.theta_o.grad is None
            assert block.ttt_head.inner_learning_rate_params.weight.grad is None
        assert self.lm_head.weight.grad is None

        loss.backward()

        wandb.log({"w_norm": self.ttt_blocks[0].ttt_head.inner.w.weight.norm()})
        wandb.log(
            {"inner_lr_params_grad": self.ttt_blocks[0].ttt_head.inner_learning_rate_params.weight.grad.norm()})
        wandb.log({"inner_lr_params": self.ttt_blocks[0].ttt_head.inner_learning_rate_params.weight.norm()})

        # NOTE: Uncomment if you want to visualize the computation graph. You
        # will need to get creative for the inner graph as it doesn't use
        # backwards or optimizer.
        # dot = make_dot(loss, params=dict(self.named_parameters()))
        # dot.render('model_graph_outer', format='png')
        # input()

        assert list(self.encoder.parameters())[0].grad is not None
        for block in self.ttt_blocks:
            assert block.ttt_head.inner.w.weight.grad is None
            assert block.ttt_head.theta_k.grad is not None
            assert block.ttt_head.theta_q.grad is not None
            assert block.ttt_head.theta_v.grad is not None
            assert block.ttt_head.theta_o.grad is not None
            assert block.ttt_head.inner_learning_rate_params.weight.grad is not None
        assert self.lm_head.weight.grad is not None
        self.optim.step()
        self.optim_inner_lr.step()
        wandb.log(
            {"inner_learning_rate_params_specific_weight":
                self.ttt_blocks[0].ttt_head.inner_learning_rate_params.weight[0][0]})

        return loss.item()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    wandb.init(
        project="ttt",
        config={
            "architecture": "Recurrent-FF",
            "dataset": "SequentialNumbers",
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model = TTTModel(
        num_layers=3, mini_batch_size=MINIBATCH_SIZE, embedding_dim=EMBEDDING_DIM, filter_dim=LOW_PASS_FILTER_DIM,
        ttt_outer_learning_rate=TTT_OUTER_LEARNING_RATE, ttt_base_inner_learning_rate=TTT_BASE_INNER_LEARNING_RATE,
        ttt_inner_learning_rate_learning_rate=TTT_INNER_LEARNING_RATE_LEARNING_RATE, vocab_size=VOCAB_SIZE,
        dropout=DROPOUT)
    model = model.to(device)

    dataset = SequentialNumbers(num_samples=NUM_SAMPLES, sequence_len=SEQUENCE_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # train
    model.train()
    for epoch in range(0, EPOCHS):
        for i, data in enumerate(dataloader):
            print(f"Epoch: {epoch}, Batch: {i} / {len(dataloader)}")
            data = data.to(device)
            data = data.permute(1, 0, 2)
            assert data.shape == (SEQUENCE_LEN, BATCH_SIZE, VOCAB_SIZE)

            # reshape to get argmax of feature dim
            data = data.argmax(dim=2)
            assert data.shape == (SEQUENCE_LEN, BATCH_SIZE)

            loss = model.train_model(data)

    # dummy eval on one sequence
    model.eval()
    for i, data in enumerate(dataloader):
        data = data.to(device)
        data = data.permute(1, 0, 2)
        assert data.shape == (SEQUENCE_LEN, BATCH_SIZE, VOCAB_SIZE)

        data = data.argmax(dim=2)
        assert data.shape == (SEQUENCE_LEN, BATCH_SIZE)

        # batch size 1
        data = data[:, 0:1]
        BATCH_SIZE = 1
        print(data)

        output = model(data)

        # just print the prediction
        output = output.argmax(dim=2)
        output = output.squeeze(1)
        print(output)

        break
