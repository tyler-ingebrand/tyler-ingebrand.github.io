---
layout: post
title:  "Zero-Shot Reinforcement Learning via Function Encoders"
date:   2024-05-05
title_size: "50px"
author: Tyler Ingebrand, Amy Zhang, Ufuk Topcu
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
        Although reinforcement learning (RL) can solve many challenging sequential decision making problems, achieving zero-shot transfer across related tasks remains a challenge. The difficulty lies in finding a good representation for the current task so that the agent understands how it relates to previously seen tasks. To achieve zero-shot transfer, we introduce the function encoder, a representation learning algorithm which represents a function as a weighted combination of learned, non-linear basis functions. By using a function encoder to represent the reward function or the transition function, the agent has information on how the current task relates to previously seen tasks via a coherent vector representation. Thus, the agent is able to achieve transfer between related tasks at run time with no additional training. We demonstrate state-of-the-art data efficiency, asymptotic performance, and training stability in three RL fields by augmenting basic RL algorithms with a function encoder task representation.
    </div>
</body>

In this paper, we seek an algorithm for *zero-shot* RL which can be applied to multi-task RL, multi-agent RL, and hidden-parameter MDPs. To do so, we introduce **the function encoder**, an algorithm for learning basis functions over arbitrary Hilbert spaces. By learning basis functions over the relevant context, (e.g. the space of reward functions, the space of transition functions, the space of adversary policies), we are able to reproduce the relevant function as a linear combination of the basis functions. Furthermore, since the coefficients of the basis functions reproduce the function of interest, they are also a fully-informative representation of the context. Thus, we pass these coefficients to the RL algorithm as a perfect context representation. 

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-RL-project-page/feWorkflow.png" width="70%" style="max-width: 70%;">
</div>

<br>

The key to this approach is the Monte-Carlo approximation of the inner product,

$$c_i = \langle f, g_i \rangle = \int_\mathcal{X} f(x)g_i(x)dx.\approx \frac{V}{N} \sum_{x_j, f(x_j) \in D}^N f(x_j) g_i(x_j),$$ 

which allows us to compute the coefficients $$c_1, c_2, ..., c_k$$ in only a few milliseconds. This enables zero-shot transfer as a small dataset $$D$$ describing a function $$f$$ can be instantaneously converted into a representation of $$f$$, $$c_f$$, which can then be used to reproduce $$f$$ for new inputs, or as a representation for downstream tasks. 

## **Zero-Shot Dynamics Predictions for Hidden-Parameter Half-Cheetah**

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-RL-project-page/hipmdp.png" width="70%" style="max-width: 70%;">
</div>

<br>

Our algorithm (FE and FE + MLP) outperforms transformers at dynamics predictions. We also get a smooth representation space with respect to a change in the hidden parameters, as shown by the cosine similiarity for dynamics under varying hidden parameters:

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-RL-project-page/hipmdp_rep.png" width="100%" style="max-width: 100%;">
</div>


## **Exploiting Adversaries in Multi-Agent Tag**

<div class="image-container" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-RL-project-page/marl.gif" width="35%" style="max-width: 35%; border: 2px solid black;">
        <img src="{{ site.baseurl }}/data/zero-shot-RL-project-page/marl.png" width="50%" style="max-width: 50%;">
</div>

By passing a function encoder representation of the adversary's policy, the RL algorithm is able to exploit each adversary specifically. As we see in the video, our agent (blue) is able to catch the adversary, and even trick the adversary into not moving by jittering around its position. As shown in the figure, our approach demonstrates significant improvements in asymptotic performance and stability relative to baselines.   

## **Multi-Task Ms. Pacman**
<div class="image-container" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-RL-project-page/atari.gif" width="35%" style="max-width: 35%; border: 2px solid black;">
        <img src="{{ site.baseurl }}/data/zero-shot-RL-project-page/mtrl.png" width="50%" style="max-width: 50%;">
</div>

<br>


By passing a function encoder representation of the reward function, our agent is able to achieve numerous tasks in Ms. Pacman using a single policy. Our agent outperforms relevant baselines in asymtotic performance and convergence speed. 
Once again, we also see a smooth, easy to work with representation space: 

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/zero-shot-RL-project-page/mtrl_rep.png" width="100%" style="max-width: 100%;">
</div>

This likely explains the improved performance of our algorithm as a smooth representation space is extremely easy to work with. 

[arxiv]: https://arxiv.org/abs/2401.17173
[code]: https://github.com/tyler-ingebrand  
