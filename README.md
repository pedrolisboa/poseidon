<!-- ![poseidon_logo](assets/logo.png) -->
<img src="assets/logo.png" alt="description of image" width="70%">



## Installation

To get started, follow the steps below.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/pedrolisboa/poseidon.git](https://github.com/pedrolisboa/poseidon.git)
    cd your-repository
    ```

## **Set Up Your Environment**

You have two options for setting up your project environment. You can either create a local virtual environment on your machine or build and run the project inside a Docker container.

---

### **Option 1: Create a Local Virtual Environment**

This is a great choice for keeping project dependencies isolated from your system's Python installation.

* **On macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
* **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

---

### **Option 2: Build and Run with Docker**

Using Docker encapsulates the entire environment, ensuring it works consistently everywhere. This is the recommended approach for avoiding "it works on my machine" issues.

1.  **Build the Docker image:**
    ```bash
    docker build -t your-image-name .
    ```
2.  **Run the container and access its shell:**
    ```bash
    docker run -it --rm your-image-name bash
    ```

3.  **Install Dependencies**
    Install the package using pip.
    ```bash
    pip install .
    ```