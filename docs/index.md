---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: home
---

<div class="image" style="text-align:center">
        <img src="{{ site.baseurl }}/data/headshot.jpg" width="50%" style="max-width: 50%;">
</div> 
<br />

Hello, and welcome to my site! I am a third year PhD student at the University of Texas at Austin working with Dr. Ufuk Topcu. I previously received my BS in computer engineering from Iowa State University. My recent research is in reinforcement learning and transfer learning. In particular, I have been applying the logic of Hilbert spaces to machine learning problems, and I have found it to be an excellent approach to achieving transfer in all sorts of problems!

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Projects</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .project-list {
            display: block;
        }
        .project-item {
            display: flex; /* Enables horizontal layout */
            align-items: center; /* Aligns items vertically */
            width: 100%; /* Ensures items take full width */
            border: 1px solid #ccc;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom:15px;
        }
        .project-item img {
            width: 240px; /* Adjust size as needed */
            object-fit: cover; /* Ensures image scales properly */
            border-radius: 8px;
            margin-right: 15px; /* Creates space between image and text */
        }
        .project-content {
            flex: 1; /* Allows text to take remaining space */
        }
        .project-item h3 {
            margin: 0 0 10px;
        }
        .project-item p {
            color: #555;
            margin: 0;
        }
        .project-item a {
            text-decoration: none;
            color: #007BFF;
            font-weight: bold;
        }
        .project-item a:hover {
            text-decoration: underline;
        }
        /* Media query for screens smaller than 768px (phones) */
        @media (max-width: 768px) {
        .project-item {
            flex-direction: column; /* Stack the text below the image on phones */
            align-items: flex-start; /* Align text to the left */
        }
        .project-item img {
            margin-bottom: 15px; /* Add space between the image and the text */
        }
        }
    </style>
</head>
<body>
    <h1>Selected Publications</h1>
    <div class="project-list">
        <div class="project-item">
            <img src="data/headers/ICML2025_cover_no_background.png" alt="A geometric characterization of transfer">
            <div class="project-content">
                <h3><a href="https://tyler-ingebrand.github.io/FunctionEncoderRL/">Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces [Under Review]</a></h3>
                <p> We introduce several improvements to the function encoder algorithm, prove a universal function space approximation theorem for function encoders, and demonstrate that the function encoder outperforms SOTA on several inductive transfer learning tasks.</p>
            </div>
        </div>
        <div class="project-item">
            <img src="data/headers/B2B.png" alt="A learned operator for an elastic plate under stress.">
            <div class="project-content">
                <h3><a href="https://tyler-ingebrand.github.io/OperatorFunctionEncoder/">Basis-to-Basis Operator Learning Using Function Encoders [CMAME 2024]</a></h3>
                <p>Basis-to-Basis operator learning is a novel method based on learned basis functions that achieves state-of-the-art performance in operator learning tasks.</p>
            </div>
        </div>
        <div class="project-item">
            <img src="data/headers/ZeroShotneuralODE.png" alt="A quadrotor flying to a target waypoint.">
            <div class="project-content">
                <h3><a href="https://tyler-ingebrand.github.io/NeuralODEFunctionEncoder/">Zero-Shot Transfer of Neural ODEs [NeurIPS 2024]</a></h3>
                <p>This work combines the recent advances in learned basis functions with neural ODEs,
                allowing for online transfer of learned system models at execution time without retraining.</p>
            </div>
        </div>
        <div class="project-item">
            <img src="data/headers/ZeroShotRL.png" alt="The procedure for zero-shot RL using function encoders.">
            <div class="project-content">
                <h3><a href="https://tyler-ingebrand.github.io/FunctionEncoderRL/">Zero-Shot Reinforcement Learning via Function Encoders [ICML 2024]</a></h3>
                <p>By representing the context of a reinforcement learning problem using function encoders, basic reinforcement learning algorithms can achieve excellent zero-shot transfer in multi-task, multi-agent, and hidden-parameter reinforcement learning problems.</p>
            </div>
        </div>

        
    </div>
</body>
</html>