# Synthetic Order Simulation with GAN

A project leveraging GANs to generate synthetic order data for simulating real-world payment transactions. This repository includes code for creating historical order datasets, training a GAN model to mimic order distributions, generating synthetic orders for future scenarios, and computing key fraud metrics (fraud rate, false positives, incorrect cancellations, and account takeover rate) to support risk assessment for new payment product launches.

**1. Project Overview**


**Objective**


The project aims to simulate realistic order data for testing and evaluation of payment product launches. By using a Generative Adversarial Network (GAN), the project generates synthetic orders that mimic real-world transaction data. This synthetic data is then used to compute key fraud-related metrics (such as fraud rate, false positives, incorrect cancellations, and account takeover rate) over a specified period (e.g., a 3-month launch period).


**What the Project Tries to Achieve**

• Data Simulation: Create historical-like order data with features including geographic, transactional, customer demographic, and fraud-related information.

• Modeling Realism: Train a GAN to learn the underlying distributions of the historical data so that the generated synthetic orders are statistically similar to real orders.

• Risk Metrics Evaluation: Use the synthetic orders to simulate the performance of a new payment product launch by calculating fraud metrics.

• Decision Support: Provide insights into how fraud prevention strategies and ML decision processes might perform in a new market environment before a product is launched.




**2. Project Implementation: Step-by-Step**


**Step 1: Historical Data Generation**

• Purpose: Create a baseline dataset that represents historical orders.

• Features: The dataset includes attributes such as:

• Geographical: country, billing_zipcode, shipping_zipcode

• Transactional: order_type (Digital or Retail), payment_amount, payment_method, order_date

• Product/Customer: product_family, is_prime_customer, customer_age, time_since_last_order

• Fraud Signals: fraud_status (0/1), ml_decision (flag produced by a machine learning model)

• Implementation: Use Python libraries (e.g., NumPy, Pandas) to randomly generate values for each feature, simulating a realistic order distribution.


**Step 2: Data Preprocessing**

• Purpose: Prepare data for training the GAN by encoding categorical variables and scaling numerical features.

• Techniques:

• Label Encoding: Convert categorical features (e.g., country, order type) into numerical labels.

• Scaling: Normalize numerical features (e.g., payment_amount, customer_age) to a 0–1 range using MinMax scaling.

• Outcome: A preprocessed dataset that is used as input for the GAN training process.


**Step 3: GAN Model Architecture**

• Components:

• Generator: A neural network that takes random noise as input and produces synthetic order data. The generator outputs a vector that mimics the structure of a real order.

• Discriminator: A neural network that receives either real or synthetic order data and predicts whether the input is genuine.

• Architecture Details:

• Both the generator and discriminator use dense (fully connected) layers with activation functions (e.g., ReLU for hidden layers, sigmoid for final output layers) to generate and classify data.

• Combined Model: The GAN model is set up by connecting the generator and discriminator, freezing the discriminator during generator training.


**Step 4: Training the GAN**

• Training Loop:

1. Discriminator Training: In each epoch, a batch of real data and a batch of generated (fake) data are fed into the discriminator. The loss is computed separately for real and fake data, then combined.

2. Generator Training: Noise is fed to the generator, and the discriminator’s output is used to update the generator’s weights. The goal is to make the discriminator classify generated data as real.

• Metrics: Monitor the loss values for both the generator and discriminator. Training continues for a predetermined number of epochs, with periodic logging to assess performance.


**Step 5: Generating Synthetic Orders**

• Generation Process:

• Once the GAN is trained, new noise vectors are passed through the generator to produce synthetic order data.

• The synthetic outputs are then post-processed:

• Inverse Scaling: Revert numerical feature scaling.

• Decoding: Convert encoded categorical features back to their original form.

• Rounding: Round binary outputs to create realistic fraud flags and decisions.

• Outcome: A new DataFrame containing synthetic orders that mirror the characteristics of historical data.


**Step 6: Calculating Fraud-Related Metrics**

• Metrics Computed:

• Fraud Rate: Percentage of orders flagged as fraudulent.

• False Positives: Orders incorrectly flagged as fraudulent by the ML decision (when fraud_status is 0 but ml_decision is 1).

• Incorrect Cancellations: Instances where fraudulent orders were not flagged (fraud_status is 1 but ml_decision is 0).

• Account Takeover Rate: A proxy metric derived from the proportion of orders with ml_decision flags among non-fraudulent orders.

• Implementation: Write functions to compute these metrics on the synthetic dataset generated for a three-month period.

**3. Conclusion****


This prototype demonstrates an end-to-end approach to:

• Simulating historical order data.

• Training a GAN model to learn and reproduce realistic order distributions.

• Generating future synthetic orders for the purpose of testing and validation.

• Calculating key fraud metrics that inform decision-making for new payment product launches.
