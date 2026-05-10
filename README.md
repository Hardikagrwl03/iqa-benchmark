# iqa-benchmark
Testing IQA Methods for both aesthetic and technical Image Quality Analysis.


### Reviews:

- NIMA (image-quality-analysis)
	- Strengths: Good at detecting focus/blur related quality issues.
    - Weakness: Inaccurate on exposure and ISO
	- Model size (approx): 25–35 MB
	- Inference time (approx): 20–50 ms per image (GPU, depends on hardware; CPU slower)
	- Backbone architecture: Inception-v2 (commonly used) / lightweight CNN variants

- LAR-IQA
	- Strengths: Robust at focus assessment and local artifact detection.
    - Weakness: Inaccurate on exposure and ISO
	- Model size (approx): 80–120 MB
	- Inference time (approx): 30-70 ms per image (GPU; CPU dependent)
	- Backbone architecture: ResNet-50 style residual CNN

- hyperIQA
	- Strengths: Strong at focus/blur detection and also performs well on exposure-related issues.
    - Weakness: Weak with ISO
	- Model size (approx): 10–40 MB (often lightweight configurations)
	- Inference time (approx): 15–40 ms per image (GPU; very efficient variants available)
	- Backbone architecture: EfficientNet / lightweight CNN with hypernetwork-style head

Notes: sizes and timings are approximate and vary by implementation, input resolution, and hardware.
