---
layout: post
title:  "Zero-Shot Transfer of Neural ODEs"
date:   2024-06-29
author: Tyler Ingebrand, Adam Thorpe, Ufuk Topcu
---


<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Abstract</title>
    <style>
        .light-gray-section {
            background-color: #e0e0e0; /* Light gray color */
            padding: 20px;
            margin: 10px;
            border-radius: 5px; /* Optional: to add rounded corners */
        }
    </style>
</head>
<body>
    <div class="light-gray-section">
        <h2 class="section-title">Abstract</h2>
        Autonomous systems often encounter environments and scenarios beyond the scope of their training data, which underscores a critical challenge: the need to generalize and adapt to unseen scenarios in real time. This challenge necessitates new mathematical and algorithmic tools that enable adaptation and zero-shot transfer. To this end, we leverage the theory of function encoders, which enables zero-shot transfer by combining the flexibility of neural networks with the mathematical principles of Hilbert spaces. Using this theory, we first present a method for learning a space of dynamics spanned by a set of neural ODE basis functions. After training, the proposed approach can rapidly identify dynamics in the learned space using an efficient inner product calculation. Critically, this calculation requires no gradient calculations or retraining during the online phase. This method enables zero-shot transfer for autonomous systems at runtime and opens the door for a new class of adaptable control algorithms. We demonstrate state-of-the-art system modeling accuracy for two MuJoCo robot environments and show that the learned models can be used for more efficient MPC control of a quadrotor.
    </div>
</body>

[In this paper (under review)][arxiv], we extend function encoders to incorporate the benefits of neural ODEs. I won't reintroduce the theory behind function encoders here, so if you are not familiar, see [the blog post]({{ site.baseurl }}{% link _posts/2024-05-05-zero-shot-RL.markdown %}) or [the paper][first paper]. 

### Why Neural ODEs?

[In the function encoder paper][first paper], we demonstrated that function encoders can predict dynamics in hidden-parameter Markov decision processes (HiP-MDPs) from a small amount of online data. HiP-MDPs describe the setting where some agent (autonomous vehicle, robot, etc) does not know everything about its environment, but it needs to act right away with only a small amount of data to guide its decision making. For example, an autonomous vehicle driving on an unknown surface, such as ice, needs to predict how it's actions (steering, braking) will affect the future state of the vehicle even though it may never have driven on ice before. By use of a function encoder, this autonomous vehicle could use a small amount of data to compute the coefficients of learned basis functions, which then allows it to predict the future state of the vehicle in a zero-shot manner. 

The first paper uses vanilla neural networks as the basis functions. In the past, neural networks have been shown to be *decent* at predicting dynamics, but inferior to other approaches such as neural ODEs. Neural ODEs use a neural network to predict the gradient of the transition function with respect to time, and then use an integrator to map a current state and action to a future state in time. This additional structure is what makes neural ODEs superior to neural networks for dynamics predictions. Real-life is inherently continuous time, and so it makes sense that we learn a continuous time model, and use an integrator to solve for discrete time points. 
The benefit of neural ODEs is orthogonal to the benefit of function encoders, and so the question is, can these two approaches be combined? In other words, can we achieve both accurate long-horizon prediction and zero-shot transfer? 

### Technical Details

The paper includes brief introductions of neural ODEs and function encoders, as well as a in-depth explanation of the mathematics of combining them. In practice, there are only two major steps.

First, the data provided for training needs to be in the form of $$ (x, u, \Delta t, \Delta x)$$ pairs, where $$\Delta x $$ is the change in state from the current state and action to the next state after $$\Delta t$$ seconds. This is because neural ODEs are expressed as $$ x_{t+\Delta t} = x_{t} + \int_0^{\Delta t} f(x_t, u_t) dt $$, where $$f$$ is the continuous time transition function. Since we are assuming $$x_t$$ is known, there is no sense in learning it, and so we only need to learn the integral term. Therefore, we train our neural ODE to reproduce the $$\int_0^{\Delta t} f(x_t, u_t) dt $$ term, which corresponds to $$ \Delta x $$.

Second, the basis functions must be separate neural ODEs. In practice, this only requires us to swap out the model architecture/forward pass method of the basis functions. This change is relatively simple, assuming the code is modular, but switches the function encoder from using a neural network as a basis function to using a neural ODE as a basis function. For a reference on good code design for the function encoder, [see here][modular code]. 

### Examples

Those familiar with neural ODEs are surely aware of the [Van Der Pol system][vdp], a toy example illustrating a continuous time system with one fixed (and possible hidden) parameter $$\mu$$. The Van Der Pol system is a two-dimensional problem, with no actions, with dynamics specified by:

$$\dot x_1 = x_2$$

$$\dot x_2 = \mu (1- {x_1}^2) x_2 - x_1$$

By varying $$\mu$$, the dynamics of the system change, and thus $$\mu$$ can be considered a hidden parameter. The goal is to predict the dynamics for any value of $$\mu$$, given only a small amount of data from the current system:

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-node/vdp_node.PNG" width="auto">
</div>

As you can see, the vanilla neural ODE can only predict a single system, and so its prediction is fixed regardless of $$\mu$$. In contrast, the FE + Neural ODE combination can accurately predict the Van Der Pol system over long time horizons from a small amount of data.

Due to the inherent scaling of neural networks, function encoders, and neural ODEs, we can apply this to much more complicated problems such as MuJoCo robotics. By comparing the prediction accuracies of numerous approaches, we can again see the benefits of combining these approaches:

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-node/mujoco_node.PNG" width="auto">
</div>

Lastly, the learned model is fully differentiable and runs at approximately the same speed as a neural ODE (since each basis function can be run in parallel). Thus, the learned model can be used for downstream tasks such as model predictive control. When applying this to a quadrotor carrying various weights, we see that the improved accuracy of the FE + Neural ODE model shows up in improved performance of downstream tasks. For example, the control of a quadrotor in the presence of hidden parameters is less efficient, and requires more course corrections, if the model does not account for hidden-parameters: 

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-node/qual_drone.PNG" width="auto">
</div>

This can be observed quantitatively as well, where the slew-rate (IE how often the actions change to course correct) is much lower for the FE + Neural ODE method. 

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-node/quan_drone.PNG" width="auto">
</div>

### The Big Picture

There are two main takeaways from this paper. First off, it is indeed possible to get accurate long horizon predictions in the presence of hidden-parameters in a zero-shot manner. This is extremely useful for any model-based control algorithms, and we are already working on followup works applying this method to robotics tasks. Secondly, the properties of the basis functions are inherited by the function encoder. In other words, if your problem has been shown to greatly benefit from some specific model architecture, such as dynamics predictions benefiting from neural ODEs, it is possible inherent those benefits while still achieving zero-shot transfer by using that architecture for the basis functions. I expect this trend to hold more broadly for all sorts of problems, and I am excited to see how combining the function encoder with all sorts of architectures (RNNs, transformers, GNNs, etc) yields fascinating results. 


[arxiv]: https://arxiv.org/abs/2405.08954
[code]: https://github.com/tyler-ingebrand  
[first paper]: https://arxiv.org/abs/2401.17173
[modular code]: https://github.com/tyler-ingebrand/FunctionEncoder\
[vdp]: https://en.wikipedia.org/wiki/Van_der_Pol_oscillator