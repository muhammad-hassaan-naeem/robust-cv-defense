from setuptools import setup, find_packages

setup(
    name="robust_cv_defense",
    version="1.0.0",
    author="Muhammad Hassaan Naeem",
    description="Adversarial Defense Pipeline with Dual-Metric Evaluation (Classical + ZKP-Informed)",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").read().splitlines(),
)
