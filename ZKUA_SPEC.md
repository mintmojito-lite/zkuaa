# ZKUA Research Report & Specification

## 1. Core Principles: The "Trust Trilemma"
The system must strictly enforce:
1.  **Privacy**: Data never leaves the client.
2.  **Utility**: The model update provides a guaranteed accuracy improvement (or meets a threshold).
3.  **Verifiability**: A Zero-Knowledge proof guarantees the computation was done correctly on the committed data.

## 2. Security Requirements

### 2.1 Commitment (The "Binding" Property)
- **Problem**: "Statistical Cheating" attack where a client changes the dataset to pass accuracy checks.
- **Requirement**: The validation dataset `D_test` must be hashed: `H = SHA256(D_test)`.
- **Implementation**:
    - This hash `H` must be computed *before* training.
    - `H` must be a **Public Input** to the ZK Circuit.
    - The verification contract/logic must check this public input against the client's registered dataset hash.

### 2.2 Circuit Logic & Constraints
The ZK circuit (or EZKL configuration) must explicitly constrain:
1.  **Accuracy Check**: `accuracy >= threshold`.
    - The threshold should be hardcoded or a public input.
    - Failing this check must prevent proof generation (or make the proof invalid).
2.  **Model Poisoning Prevention**: `||W_new - W_old|| <= max_norm`.
    - This ensures the update doesn't drastically destabilize the global model.

### 2.3 Model Architecture ("BiggerNet")
To ensure proof generation times are feasible (Efficiency), the model must match this specific architecture:
-   **Input Layer**: 20 neurons
-   **Hidden Layer 1**: 64 neurons (ReLU)
-   **Hidden Layer 2**: 32 neurons (ReLU)
-   **Hidden Layer 3**: 16 neurons (ReLU)
-   **Output Layer**: 2 neurons (Softmax/Logits)
-   *Constraint*: Deeper networks will exceed the < 5s proof generation target.

## 3. Benchmarking Targets ("The 5/10/2 Rule")
To be viable for federated learning cycles, the system must meet:
-   **Proof Generation Time**: < 5 seconds (on CPU).
-   **Verification Time**: < 10 ms.
-   **Proof Size**: < 2 KB.

## 4. Attack Vectors to Mitigate
-   **Attack A (The Lazy Hospital)**: Sending noise instead of training. Mitigated by Accuracy Check.
-   **Attack B (Data Switching)**: Swapping test data for easy samples. Mitigated by Dataset Hashing (Commitment).
-   **Attack C (Model Poisoning)**: Sending malicious weights. Mitigated by Norm Check.
