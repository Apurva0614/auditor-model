# Architecture: Auditor Model System

## Why Entropy and Margin Outperform Raw Confidence

A naive auditor might flag samples purely based on the primary model's maximum predicted probability. However, raw confidence (the maximum class probability) is insufficient because a model can be confidently wrong — particularly near decision boundaries where many classifiers produce overconfident predictions on unseen data.

Entropy provides a global view of uncertainty across the full probability distribution. A model that distributes probability mass more evenly across multiple classes will have high entropy even if its peak probability is moderate. This catches cases where the model is genuinely undecided between several options, not just two.

Margin measures the gap between the top-ranked and second-ranked class probabilities. A small margin means the model barely preferred one class over another, which is a strong signal of unreliability. Importantly, a model can have a high peak probability in an overconfressed binary setting yet still have a narrow margin when the two leading classes are close.

Together, entropy and margin form complementary views: entropy is sensitive to many-class ambiguity, while margin captures binary-like uncertainty even in the multiclass regime. Neither signal alone dominates the other, so the auditor learns a richer function of uncertainty than any single scalar threshold on raw confidence could provide.

## The Training Data Contamination Problem

The most critical design constraint of the auditor is that it must never be trained on the same data used to train the primary model. The reason is subtle but important.

During training, the primary model memorises many details of the training distribution. Its error pattern on training data does not reflect how it behaves on novel inputs. On training samples the primary has seen, it will produce low-entropy, high-confidence, wide-margin predictions even for inputs it would misclassify during deployment. Consequently, a contaminated auditor would learn that high confidence means error-free, which is exactly backward from its true behaviour on held-out data.

By using a held-out validation set, the auditor is exposed to authentic primary-model errors: cases where the primary produces a confident prediction that nonetheless disagrees with the true label. These are the operationally relevant failure modes and the ones the auditor needs to learn to detect.

This is analogous to the split-conformal validity guarantee in conformal prediction: the calibration set must be exchangeable with the deployment distribution and independent of model training.

## Threshold Calibration Strategy and the Precision-Recall Tradeoff

The suppression threshold tau controls the tradeoff between two objectives: catching real errors (recall) and avoiding unnecessary suppression of correct predictions (precision). A low tau suppresses aggressively, catching most errors but also suppressing many correct AI outputs and wasting human review capacity. A high tau is conservative and suppresses rarely, preserving AI throughput but missing errors.

The router's sweep_thresholds function evaluates the joint accuracy across a grid of tau values. Joint accuracy is defined as the weighted combination of AI accuracy on shown cases and human accuracy on suppressed cases. The goal is to find the tau that maximises joint accuracy gain over AI-only deployment.

The optimal tau depends on the ratio of human accuracy to AI accuracy and on the base error rate of the primary model. If human accuracy is low, aggressive suppression is harmful even if the auditor is well-calibrated. If the AI error rate is high, a lower tau is justified to capture more errors. In practice, the best_threshold method searches this space automatically and sets the threshold accordingly.

For high-stakes settings, it is advisable to set tau based on a recall target rather than accuracy gain, ensuring that a minimum fraction of true errors are caught regardless of precision costs.

## Extending to a New Domain or Real Dataset

To adapt this system to a new problem, four changes are required.

First, replace the synthetic data generation in main.py with a call to load_dataset pointing at a real CSV file. The last column of the CSV is treated as the target label, and all preceding columns are features. Preprocessing steps such as encoding categorical variables or imputing missing values should be applied before passing data to the system.

Second, choose the primary model type that best suits the problem. Gradient boosting typically performs best on tabular data with mixed feature types. Logistic regression is a good baseline and is the most interpretable. Random forest provides robustness with minimal hyperparameter tuning.

Third, adjust the auditor threshold to reflect the actual human accuracy in your domain. A radiologist reviewing suppressed images may have 90 percent accuracy, not 72 percent, which shifts the optimal tau considerably.

Fourth, if the classification task is multiclass rather than binary, the roc_auc computation in PrimaryModel.evaluate automatically switches to one-vs-rest averaging, and the auditor features continue to work without modification because entropy and margin generalise naturally to any number of classes.
