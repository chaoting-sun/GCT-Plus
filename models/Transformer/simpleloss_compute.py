
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, loss_function, optim):
        self.generator = generator
        self.loss_function = loss_function
        self.optim = optim

    def __call__(self, x, y, norm):

        x = self.generator(x)

        loss = self.loss_function(x.contiguous().view(-1, x.size(-1)), # why???
                                  y.contiguous().view(-1)) / norm

        if self.optim is not None:
            loss.backward()
            self.optim.step()
            self.optim.optimizer.zero_grad()

        return loss.data * norm
