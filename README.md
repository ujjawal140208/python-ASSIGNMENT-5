# python-ASSIGNMENT-5
# Campus Energy-Use Dashboard (Capstone Project)

## 1. Objective

This project is a capstone assignment for the course **Programming for Problem Solving using Python**.  
The goal is to build an **end-to-end pipeline** that reads raw electricity meter data for campus buildings, analyzes it, and produces a **visual dashboard** and a **text-based executive summary** that can help the administration make energy-saving decisions.

## 2. Dataset

- Input data is placed in the `data/` folder.
- Each CSV file represents **one building's** energy usage for a given period (e.g., one month).
- Expected columns in each CSV:

  - `timestamp` – Date and time of the reading (e.g., `2024-01-01 00:00:00`)
  - `kwh` – Energy consumed in that interval (float)
  - `building` – (Optional) If missing, the building name is inferred from the filename.
