import torch

def first_order_loss(neural_network, differential_equation, ic):

    def trial_solution(x):
        result = ic + x * neural_network(x)

        return result
    
    delta = 1e-5
    x = torch.linspace(0, 1, 10, requires_grad=True).unsqueeze(1)
    dtrial_dx = (trial_solution(x+delta) - trial_solution(x)) / delta
    individual_error = (dtrial_dx - differential_equation(x))**2
    # individual_error = torch.abs(dtrial_dx - differential_equation(x))

    return torch.sum(individual_error)


