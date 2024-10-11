
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curve(scores, plot_path):
    plt.figure()
    auc_scores = scores['auc_scores']
    plt.plot(range(len(auc_scores)), auc_scores, label='AUC Score')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(plot_path)

def main():
    # Load model results
    results_path = '../models/model_results.csv'
    scores = pd.read_csv(results_path)
    
    # Path to save the plot
    plot_path = '../plots/learning_curve.png'
    
    # Plot and save the learning curve
    plot_learning_curve(scores, plot_path)

if __name__ == '__main__':
    main()
