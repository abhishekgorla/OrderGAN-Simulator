# Synthetic Order Simulation with GAN

**Overview**

A project leveraging GANs to generate synthetic order data for simulating real-world payment transactions. This repository includes code for creating historical order datasets, training a GAN model to mimic order distributions, generating synthetic orders for future scenarios, and computing key fraud metrics (fraud rate, false positives, incorrect cancellations, and account takeover rate) to support risk assessment for new payment product launches.

**Steps in the Project**


**Step 1: Create Historical Data**


A function create_historical_data is used to generate a dataset that mimics historical transaction records. The dataset includes:

• Categorical Features:

• country: A list of countries (US, UK, CA, AU)

• order_type: Type of order (Digital, Retail)

• payment_method: Payment method used (Credit Card, Debit Card, PayPal, Bank Transfer)

• product_family: Category of product purchased (Electronics, Clothing, Home, Books)

• billing_zipcode, shipping_zipcode: Random zip codes

• Numerical Features:

• payment_amount: The transaction amount

• customer_age: Customer’s age

• time_since_last_order: Time elapsed since the last order

• Target Features:

• fraud_status: Binary flag indicating if the order is fraudulent (0 = no fraud, 1 = fraud)

• ml_decision: The fraud detection model’s decision (0 = not fraud, 1 = fraud)


The function uses random sampling methods to create a dataset with n_samples rows, and the resulting DataFrame is returned.


**Step 2: Preprocess the Data**


The data is preprocessed by encoding categorical features and scaling numerical features. This ensures that the data is in a format suitable for training the GAN:

• Categorical Features: Encoded using LabelEncoder from scikit-learn.

• Numerical Features: Scaled using MinMaxScaler to ensure all values are between 0 and 1.

• Target Features: The binary target features fraud_status and ml_decision are retained in the dataset for later analysis.


**Step 3: Build the GAN Model**


A GAN model is built consisting of two parts:

1. Generator: This neural network takes random noise as input and generates synthetic transaction data. It is designed to output data in the same dimensional space as the historical dataset, ensuring it has the same features (e.g., payment_amount, country, etc.).

• The generator consists of dense layers, with the final output layer having a sigmoid activation function to keep the generated values within the range of [0, 1].

2. Discriminator: This neural network takes real or generated order data as input and outputs the probability that the data is real (as opposed to fake). The discriminator is trained to distinguish between real and generated data.

• It consists of dense layers, with the final output layer using a sigmoid activation to predict if the input data is real or fake.


The combined GAN model trains the generator by freezing the discriminator during generator training, so the generator can learn to produce increasingly realistic data.


**Step 4: Train the GAN**


The GAN is trained for a specified number of epochs. During each epoch:

1. The discriminator is trained on a batch of real data and a batch of fake (generated) data.

2. The generator is then trained using the combined model (discriminator is frozen) to improve its ability to generate realistic synthetic data.


The loss of both the generator and discriminator is tracked during training. The training process includes regular outputs showing the progress in terms of loss and accuracy.


**Step 5: Generate New Synthetic Orders**


Once the GAN is trained, the generate_synthetic_orders function is used to generate synthetic orders. This function:

1. Takes random noise as input for the generator.

2. Generates synthetic order data.

3. Inversely scales the numerical features back to their original scale.

4. Reverses the label encoding for categorical features to obtain meaningful labels.

5. Rounds the binary target features (fraud_status, ml_decision) to the nearest integer.


The result is a new DataFrame containing synthetic orders, which can be used for fraud detection testing.


**Step 6: Generate Fraud Metrics for New Payment Product Launch**


The synthetic orders are analyzed to generate key fraud metrics, including:

• Fraud Rate: Percentage of orders flagged as fraud (fraud_status == 1).

• False Positives: Orders that are not fraud but were flagged by the ML model (fraud_status == 0 and ml_decision == 1).

• Incorrect Cancellations: Orders that are fraudulent but were not flagged by the ML model (fraud_status == 1 and ml_decision == 0).

• Account Takeover Rate: Fraction of orders flagged as fraud (ml_decision == 1) that are actually not fraudulent (fraud_status == 0), representing potential account takeover incidents.


These metrics help assess the performance of the fraud detection model and provide insights for future product launches.

