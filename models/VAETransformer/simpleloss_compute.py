
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, loss_function, optim):
        self.loss_function = loss_function
        self.optim = optim

    def __call__(self, x, y, norm, mu, logvar):        
        rec_loss, KL_div = self.loss_function(x.contiguous().view(-1, x.size(-1)), 
                                              y.contiguous().view(-1), mu, logvar)
        loss = rec_loss / norm + KL_div

        if self.optim is not None:
            loss.backward()
            self.optim.step()
            self.optim.optimizer.zero_grad()

        return rec_loss.data, KL_div
