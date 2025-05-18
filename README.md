# Maglev Train Simulation using Python & OpenGL

A 3D simulation of magnetic levitation (maglev) train physics using Python and OpenGL.

## Overview

This project simulates the magnetic levitation physics of a maglev train system, visualizing the electromagnetic forces, train movement, and track interactions in a 3D environment.

## Features

-   Real-time 3D visualization of maglev train dynamics
-   Physics simulation of electromagnetic levitation forces
-   Interactive camera controls to view the simulation from different angles
-   Adjustable parameters for magnetic field strength, train mass, speed etc.
-   Track editor to create custom maglev routes

## Requirements

-   Python 3.7+
-   PyOpenGL
-   NumPy
-   PyGame (for window management)

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github/com/Dqrshan/maglev-simulation.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd maglev-simulation
    ```

3.  Virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

5.  Run the simulation:
    ```bash
    python maglev.py
    ```
