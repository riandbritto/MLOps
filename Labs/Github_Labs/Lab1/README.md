# Lab 1 – MLOps Basics

## Overview

In this lab, I worked on understanding the basic workflow of an MLOps project. The main goal was to set up a clean Python project, write simple functions, test them using different frameworks, and automate the testing process using GitHub Actions.

---

## What I Did

* Created a virtual environment to isolate dependencies
* Organized the project into `src` and `test` folders
* Implemented a simple `calculator.py` module with basic arithmetic functions
* Wrote test cases using:

  * **Pytest** for simple function-based testing
  * **Unittest** for structured class-based testing
* Set up **GitHub Actions** to automatically run tests whenever I push code

---

## Project Structure

```text
Lab1/
├── src/
│   └── calculator.py
├── test/
│   ├── test_pytest.py
│   └── test_unittest.py
├── .github/workflows/
├── requirements.txt
└── README.md
```

---

## How to Run the Lab

### 1. Go to the Lab1 folder

```bash
cd Labs/Github_Labs/Lab1
```

### 2. Create and activate virtual environment

```bash
python -m venv lab_01
source lab_01/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run tests

```bash
pytest test/test_pytest.py
python -m unittest test.test_unittest
```

---

## GitHub Actions

I configured workflows inside `.github/workflows/` so that both pytest and unittest run automatically whenever I push changes to the repository. This helped me understand how CI/CD works in practice.

---

## Challenges Faced

Initially, I faced issues with running tests from the wrong directory and incorrect folder structure. I resolved this by ensuring I was inside the `Lab1` folder and placing workflow files in the correct `.github/workflows/` directory.

---
## GitHub Actions Result

After setting up the workflows, I verified that the pipelines were triggered successfully on pushing code to the repository. Both pytest and unittest workflows ran without errors, confirming that the CI/CD setup is working correctly.

Below is a screenshot showing the successful execution of GitHub Actions:

![GitHub Actions Success](images/actions_success.png)


## Conclusion

This lab helped me understand the importance of project structure, testing, and automation in MLOps. Even though the example was simple, it gave me a clear idea of how real-world ML projects maintain code quality using CI/CD pipelines.

Name: Rian Renold Dbritto
NUID:002026598
