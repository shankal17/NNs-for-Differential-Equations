import torch
import torch.optim as optim

def first_order_loss_with_ic(neural_network, a, g, ic, domain_lower_bound=0, domain_upper_bound=1, num_points=10):

    def trial_solution(x):
        result = ic + x * neural_network(x)

        return result
    
    delta = 1e-5
    x = torch.linspace(domain_lower_bound, domain_upper_bound, num_points, requires_grad=True).unsqueeze(1)
    dtrial_dx = (trial_solution(x+delta) - trial_solution(x)) / delta
    individual_error = (dtrial_dx - (g(x) - a(x)*trial_solution(x)))**2
    # individual_error = torch.abs(dtrial_dx - (g(x) - p(x)*trial_solution(x)))

    return torch.sum(individual_error)

def second_order_loss_with_ic(neural_network, a, b, g, ic, ic_prime, domain_lower_bound=0, domain_upper_bound=1, num_points=10):

    def trial_solution(x):
        result = ic + x*ic_prime + (x**2) * neural_network(x)
        # print(result.requires_grad)
        return result
    
    delta = 1e-3
    x = torch.linspace(domain_lower_bound, domain_upper_bound, num_points, requires_grad=True).unsqueeze(1)
    dtrial_dx = (trial_solution(x+delta) - trial_solution(x)) / delta
    d2trial_dx2 = (trial_solution(x+delta) - 2*trial_solution(x) + trial_solution(x-delta))/(delta**2)
    individual_error = (d2trial_dx2 - (g(x) - a(x)*dtrial_dx - b(x)*trial_solution(x)))**2
    return torch.sum(individual_error)


def fit_first_order_ic_model(neural_network, a, g, ic, domain_lower_bound=0, domain_upper_bound=1, num_points=10, epochs=1000, lr=0.003):
    
    def trial_solution(x, nn):
        result = ic + x * nn(x)
        return result
    
    def loss():
        delta = 1e-5
        x = torch.linspace(domain_lower_bound, domain_upper_bound, num_points, requires_grad=True).unsqueeze(1)
        dtrial_dx = (trial_solution(x+delta, neural_network) - trial_solution(x, neural_network)) / delta
        individual_error = (dtrial_dx - (g(x) - a(x)*trial_solution(x, neural_network)))**2
        return torch.sum(individual_error)
    
    display_step = epochs // 10
    optimizer = optim.Adam(neural_network.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = loss()
        epoch_loss.backward()
        optimizer.step()
        if epoch % display_step == 0:
            print(epoch_loss.item())
    
    return trial_solution
        