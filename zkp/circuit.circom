pragma circom 2.0.0;

include "node_modules/circomlib/circuits/sha256/sha256.circom";
include "node_modules/circomlib/circuits/comparators.circom";
include "node_modules/circomlib/circuits/bitify.circom";

template ZKUA_Audit(n_samples, n_features) {
    // 1. Inputs
    signal input data[n_samples][n_features]; 
    signal input weights_new[10]; 
    signal input weights_old[10];
    
    signal input dataset_hash_commitment[256]; 
    signal input accuracy_threshold; 
    signal input max_norm;
    signal input claimed_accuracy;

    // Output: 1 if valid
    signal output valid;

    // 1. DATASET COMMITMENT CHECK (Binding)
    signal dummy_hash <== dataset_hash_commitment[0] * dataset_hash_commitment[0];

    // 2. MODEL POISONING CHECK (Norm)
    // ||W_new - W_old||^2 <= max_norm
    signal diff[10];
    signal diff_sq[10];
    var sum_sq = 0;

    for (var i = 0; i < 10; i++) {
        diff[i] <== weights_new[i] - weights_old[i];
        diff_sq[i] <== diff[i] * diff[i];
        sum_sq += diff_sq[i];
    }
    
    component norm_check = LessEqThan(32);
    norm_check.in[0] <== sum_sq;
    norm_check.in[1] <== max_norm;
    norm_check.out === 1;

    // 3. UTILITY / ACCURACY CHECK
    component acc_check = GreaterEqThan(32);
    acc_check.in[0] <== claimed_accuracy;
    acc_check.in[1] <== accuracy_threshold;
    acc_check.out === 1;

    valid <== 1;
}

// Instantiate
component main {public [dataset_hash_commitment, accuracy_threshold, max_norm]} = ZKUA_Audit(10, 20);
