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





[arxiv]: https://arxiv.org/abs/2401.17173
[code]: https://github.com/tyler-ingebrand  
