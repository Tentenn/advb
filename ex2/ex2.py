# Numpy version: 1.23.5
# Scikit-learn: 1.2.1
# Matplotlib: 3.7.1

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Function for 2.1 -> Reconstitues the data from the given numpy arrays
def reconstitute_data(residues_per_protein, encodings, labels_all):
    data = []
    start_idx = 0

    for num_residues in residues_per_protein:
        end_idx = start_idx + num_residues

        # Slice the encodings and labels arrays for the current protein
        protein_encodings = encodings[start_idx:end_idx, :]
        protein_labels = labels_all[start_idx:end_idx, :]

        # Create a dictionary for the current protein
        protein_data = {
            'encodings': protein_encodings,
            'labels': protein_labels
        }

        # Append the dictionary to the data list
        data.append(protein_data)

        # Update the start index for the next protein
        start_idx = end_idx

    return data

# Helper function for 2.2
def reorganize_data(reconstituted_data):
    X, y = [], []

    for protein_data in reconstituted_data:
        X.append(protein_data["encodings"])
        y.append(protein_data["labels"])

    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y

def eval():
    from sklearn.metrics import classification_report

    # ... (Other functions and code from previous answers should be placed here)

    # Train the best estimator on the entire training set
    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_train, y_train.ravel())

    # Make predictions on the test set
    y_pred = best_estimator.predict(X_test)

    # Calculate performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save the results to a .txt file
    with open('results.txt', 'a') as f:
        f.write("\nBest performing combination:\n")
        f.write(str(grid_search.best_params_))
        f.write("\n\nPerformance metrics on test set:\n")
        f.write("Class\tAccuracy\tPrecision\tRecall\n")

        for cls, scores in report.items():
            if cls != 'accuracy' and cls != 'macro avg' and cls != 'weighted avg':
                f.write(f"{cls}\t{scores['precision']:.3f}\t{scores['recall']:.3f}\t{scores['f1-score']:.3f}\n")

        f.write(f"\nOverall accuracy: {report['accuracy']:.3f}\n")

    print("Results saved to 'results_final.txt'")



def main():
    residues_per_protein = np.load('encodings/residues_per_protein.npy')
    encodings = np.load('encodings/encodings_all.npy')
    labels_all = np.load('encodings/labels_all.npy')
    
    reconstituted_data = reconstitute_data(residues_per_protein, encodings, labels_all)
    X, y = reorganize_data(reconstituted_data)

    # X = X[:10000]
    # y = y[:10000]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Print the shapes of the resulting sets
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Define the estimator
    estimator = MLPClassifier(max_iter=500, random_state=42)

    # Define the hyperparameters to test
    param_grid = {
        'hidden_layer_sizes': [(n,) for n in range(5, 21, 5)],
        'learning_rate_init': [0.01, 0.001, 0.0001]
    }

    ########### GRIDSEARCH ############ 
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator, param_grid, n_jobs=8)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train.ravel())

    # Save the results to a .txt file
    with open('results.txt', 'w') as f:
        for i, params in enumerate(grid_search.cv_results_['params']):
            line = f"{params}: mean_test_score={grid_search.cv_results_['mean_test_score'][i]:.3f}, std_test_score={grid_search.cv_results_['std_test_score'][i]:.3f}\n"
            f.write(line)

    print("Results saved to 'results.txt'")

    ############ TRAIN WITH BEST PARAMS #########
    # Train the best estimator on the entire training set
    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_train, y_train.ravel())

    # Make predictions on the test set
    y_pred = best_estimator.predict(X_test)

    # Calculate performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save the results to a .txt file
    with open('results_final.txt', 'a') as f:
        f.write("\nBest performing combination:\n")
        f.write(str(grid_search.best_params_))
        f.write("\n\nPerformance metrics on test set:\n")
        f.write("Class\tAccuracy\tPrecision\tRecall\n")

        for cls, scores in report.items():
            if cls != 'accuracy' and cls != 'macro avg' and cls != 'weighted avg':
                f.write(f"{cls}\t{scores['precision']:.3f}\t{scores['recall']:.3f}\t{scores['f1-score']:.3f}\n")

        f.write(f"\nOverall accuracy: {report['accuracy']:.3f}\n")

    print("Results saved to 'results_final.txt'")

    #### PLOT TRAIN AND TEST PERFORMANCE 200 EPOCHS #####
    # Set the best performing hyperparameters from the GridSearchCV result
    best_params = grid_search.best_params_

    # Create the estimator with the best performing hyperparameters, max_iter=1, and warm_start=True
    estimator = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        learning_rate_init=best_params['learning_rate_init'],
        max_iter=1,
        warm_start=True,
        random_state=42
    )

    train_accuracies = []
    test_accuracies = []

    # Train the model for 200 epochs
    for epoch in range(200):
        # Train one epoch and update the model
        estimator.fit(X_train, y_train.ravel())

        # Compute accuracy on the training set
        train_accuracy = estimator.score(X_train, y_train)
        train_accuracies.append(train_accuracy)

        # Compute accuracy on the test set
        test_accuracy = estimator.score(X_test, y_test)
        test_accuracies.append(test_accuracy)

    # Plot the performance on the training and test sets
    plt.figure()
    plt.plot(train_accuracies, label='Training set')
    plt.plot(test_accuracies, label='Test set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Performance on Training and Test Sets')

    # Save the plot as performance.pdf
    plt.savefig('performance.pdf')

    print("Performance plot saved as 'performance.pdf'")

if __name__ == "__main__":
	main()