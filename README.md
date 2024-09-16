# Correcting-Cognitive-Biases-in-Doctor-patient-Interactions

It is observed that fatigued physicians often tend to resort to heuristics under time pressure which can lead to incorrect Bayesian update.

Medical errors due to cognitive biases occur in 1.7-6.5 % of all hospital admissions causing up to 100,000 unnecessary deaths each year, and perhaps one million in excess injuries in the USA. In 2008, total cost was USA $19.5 billion. The incremental cost associated on average was about US$ 4685 and an increased length of stay of about 4.6 days.

Addressing biases that arise in data interpretation, feature selection, model evaluation, and deployment is therefore crucial. While previous work has focused on addressing biases in medical large language models, the proposed approach aims to design a personalized model for bias identification and suggestive action in doctor-patient interactions.

The methodology involves training an NLP model for bias identification using manually annotated MedQA dataset from physician-patient encounters. Disease and bias-specific corrective actions are then implemented, followed by training a neural network for Bayesian update and probability map generation.

The performance metrics for the ML model include tuning hyperparameters based on desirable thresholds, such as minimizing type 2 errors in diagnosis and increasing model sensitivity.
