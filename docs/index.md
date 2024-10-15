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
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .project-item {
            width: 300px;
            border: 1px solid #ccc;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .project-item img {
            width: 100%; /* Make the image fill the container width */
            height: 200px; /* Set a fixed height */
            object-fit: contain; /* Ensures the image scales and crops to fit */
            border-radius: 8px;
        }
        .project-item h3 {
            margin: 10px 0;
        }
        .project-item p {
            color: #555;
        }
        .project-item a {
            text-decoration: none;
            color: #007BFF;
            font-weight: bold;
        }
        .project-item a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>Selected Publications</h1>
    <div class="project-list">
        <!-- Project 2 -->
        <div class="project-item">
            <img src="data/headers/B2B.png" alt="A learned operator for an elastic plate under stress.">
            <h3><a href="https://tyler-ingebrand.github.io/OperatorFunctionEncoder/">Basis-to-Basis Operator Learning Using Function Encoders [Under Review]</a></h3>
            <p>Basis-to-Basis operator learning is a novel method based on learned basis functions that achieves state-of-the-art performance in operator learning tasks.</p>
        </div>


        <!-- Project 1 -->
        <div class="project-item">
            <img src="data/headers/ZeroShotneuralODE.png" alt="A quadrotor flying to a target waypoint.">
            <h3><a href="https://tyler-ingebrand.github.io/NeuralODEFunctionEncoder/">Zero-Shot Transfer of Neural ODEs [NeurIPS 2024]</a></h3>
            <p>This work combines the recent advances in learned basis functions with neural ODEs,
  allowing for online transfer of learned system models at execution time without retraining.</p>
        </div>

        <!-- Project 2 -->
        <div class="project-item">
            <img src="project2-cover.jpg" alt="Project 2 Cover Image">
            <h3><a href="project2.html">Project 2 Title</a></h3>
            <p>Short description of Project 2 goes here. Highlight key points and outcomes of the project.</p>
        </div>

        
    </div>
</body>
</html>