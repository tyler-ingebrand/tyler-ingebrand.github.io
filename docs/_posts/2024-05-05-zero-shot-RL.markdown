---
layout: post
title:  "Zero-Shot Reinforcement Learning Via Function Encoders"
date:   2024-05-04
---

In this paper, I proposed an algorithm for learning basis functions for arbitrary Hilbert spaces. The paper introduces this idea and uses it to achieve zero-shot transfer in hidden-parameter system identification, multi-agent RL, and multi-task RL. However, learning basis functions is far more useful than just these fields. The purpose of this post is to introduce the idea of function encoders while a more detailed discussion on RL can be found in the [paper (ICML 2024)][arxiv]. 



## What is Zero-Shot Transfer?

The entire field of machine learning is built around learning a function $$f : \mathcal{X} \to \mathcal{Y}$$, where $$\mathcal{X}$$ is the input space (IE an image, data on a user, the state of a system) and $$\mathcal{Y}$$ is the output space (IE a scalar value, a vector, a probability). Typically a dataset $$\{x_i, f(x_i)\}_{i=1}^N$$ is provided, and a neural network $$f_\theta$$ is trained via gradient descent so that $$f(x_i)=f_\theta(x_i)$$ for all i. Additionally, the hope is that $$f(x) \approx f_\theta(x)$$ for any previously unseen $$x$$. In other words, the neural network $$f_\theta$$ "reproduces" $$f$$. 

This framework allows Netflix to predict what movies a user will like, allows Cruise to predict the future state of their autonomous vehicle, allows ChatGPT to respond to your questions, and much more. However, this framework does not fully address the messiness of real life. Consider Netflix; Their goal is to predict the rating that a user will give to a particular movie. If Netflix modeled user preferences via a single neural network, they could only predict if the *average* user would like a particular movie. However, every user has different preferences and their real task is to predict if a *given* user will like a movie. Furthermore, rather than having a single dataset of $$(movie, rating)$$ pairs, they have a large number of users and for each user they have a dataset of $$(movie, rating)$$ pairs. In other words, they have a set of datasets, where each dataset corresponds to a particular user. Their objective is to predict if a particular user will enjoy a movie, given this user's past preferences for similar movies and the knowledge of what other similar users think. The way to understand this it that every user has an internal function mapping movies to ratings. The challenge of reproducing this function for a new user with limited amounts of data is called *zero-shot transfer*.

## Formalizing Transfer

Consider the set of functions $$\mathcal{F} = \{f \mid f: \mathcal{X} \to \mathcal{Y} \}$$. Suppose we have a set of datasets $$\mathcal{D} =\{ D_1, D_2, ...\} $$ where $$D_j = \{x_i, f_j(x_i)\}_{i=1}^N$$ and $$f_j \in \mathcal{F}$$. This could represent the set of datasets that Netflix would have, where each $$D_j$$ is data on a particular user. The goal is to learn some sort of model using $$\mathcal{D}$$ so that for any unseen function $$f$$, we can reproduce $$f$$ during "execution" time. Of course, this requires a dataset $$D = \{x_i, f(x_i)\}_{i=1}^N$$ to condition the model on at execution time. Returning to our Netflix example, this means Netflix wants to predict if a new user will like some movie given a small dataset of this user's past preferences. 

The first question is what form this model should have. The naive approach is to train a separate neural network for every function in $$\mathcal{F}$$. Unfortunately, there are infinite of them, so that idea is out. Therefore, we need some sort of structure on $$\mathcal{F}$$ so that a model is feasible. Suppose $$\mathcal{F}$$ is a Hilbert space, which gives us some nice properties such as the continuity of the functions and the completeness of the space. Then we can reproduce every function $$f$$ as a weighted combination of (orthonormal) basis functions, 

$$f(x) = \sum_{i=1}^k c_i g_i(x),\quad \quad \quad[1]$$

where $$g_1, g_2, ..., g_k$$ are basis functions and $$c_1, c_2, ..., c_k$$ are scalar coefficients (which are unique to $$f$$). This gives us a way of expressing the model. We simply need to learn the basis functions $$g_1, g_2, ..., g_k$$, and find a way to compute the coefficients for any function $$f$$. In theory, we may need infinite basis functions to perfectly reproduce $$f$$, but in practice we are content with approximating $$f$$ via a finite number of basis functions. 

As mentioned, we need to compute the coefficients $$c_1, c_2, ..., c_k$$ for any function $$f$$, given our basis functions. Fortunately, there is a closed form solution:

$$c_i = \langle f, g_i \rangle = \int_\mathcal{X} f(x)g_i(x)dx.\quad \quad \quad[2]$$ 

Unfortunately, this is often impractical as $$\mathcal{X}$$ is typically high-dimensional, and computing this integral is intractable. Furthermore, it would require exact knowledge of $$f$$, which defeats the purpose of modeling it. Lastly, we only have the dataset $$D$$, which gives us a few example input-output data points which are taken from $$f$$. Instead, we need to approximate this integral using our small data set. Quick side note: This definition assumes the output space is scalar. If it isn't, replace $$f(x)g_i(x)$$ in the above and below equations with $$\langle f(x), g_i(x) \rangle_\mathcal{Y}$$. 

To approximate the inner product with a dataset, we turn to Monte Carlo integration:

$$c_i = \langle f, g_i \rangle \approx \frac{V}{N} \sum_{x_j, f(x_j) \in D}^N f(x_j) g_i(x_j),\quad \quad \quad[3]$$

where $$V$$ is the volume of $$\mathcal{X}$$. Basically, we are approximating the integral with whatever data we happen to have. Given equation 1, we can reproduce any function with our basis functions. Given equation 3, we can compute any function's coefficients from a small amount of data. Thus, we are able to model the space $$\mathcal{F}$$. The only challenge is finding the basis functions, given the training set $$\mathcal{D}$$. 

Suppose we create $$k$$ neural networks, each of which is a basis function. Then, we use those learned basis functions to reproduce $$f_1, f_2, ... $$ via the data sets $$D_1, D_2, ...$$ and equations 1 and 3. Initially, these basis functions will **not** span $$\mathcal{F}$$, and so the basis functions will fail to reproduce $$f_1, f_2, ... $$. We can measure the error of this prediction with respect to a single function $$f_j$$ in the normal way (Mean squared error, cross entropy, whatever), and then we can minimize this error for all functions $$f_1, f_2, ... $$ via gradient descent. This process yields an algorithm for learning basis functions for an arbitrary space $$\mathcal{F}$$: 

1. &nbsp; Initialize basis functions $${𝑔_1,𝑔_2,…, 𝑔_𝑘}$$ parameterized by $$\theta$$
2. &nbsp; While not converged:        
3. &emsp; $$𝑙𝑜𝑠𝑠=0$$
4. &emsp; For $$D_i \in \mathcal{D}$$:
5. &emsp; &emsp; $$𝑐_𝑖 = \langle 𝑓,𝑔_𝑖 \rangle \forall 𝑖$$ 
6. &emsp; &emsp; $$f = \sum_{i=1}^k c_i g_i $$ 
7. &emsp; &emsp; $$loss = loss + \mid \mid \hat{f} - f \mid \mid ^2 $$
8. &emsp; $$\theta = \theta - \alpha \nabla_\theta loss $$

I've left the algorithm in terms of the inner product and vector addition/scalar multiplication terms to highlight it can be applied to any Hilbert space, so long as the basis functions are differentiable. For example, this is what it looks like applied to 3-dimensional Euclidean space:

<video width="1000" height="500" controls="controls">
  <source src="{{ site.baseurl }}/data/animation_2b.mp4" type="video/mp4">
  Your browser may not be able to load this video. 
</video>

The black arrows are the basis vectors, and the blue square indicates the 2-dimensional space $$\mathcal{F}$$ that we are trying to fit. The plot on the right shows the error in reproducing vectors drawn from $$\mathcal{F}$$. As you can see, the basis vectors slowly converge to spanning the space $$\mathcal{F}$$. Intuitively, the same thing is happening when applying this algorithm to function spaces, where the functions slowly move to span the functions present in the set of datasets $$\mathcal{D}$$. 



[arxiv]: https://arxiv.org/abs/2401.17173
[code]: https://github.com/tyler-ingebrand  
