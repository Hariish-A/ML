from collections import Counter
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    

class PreProcessing:

    def normalize(x):
        # Standardize features
        scaler = StandardScaler()
        return scaler.fit_transform(x), scaler
    
    def apply_pca(x, n_components=None):
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by each component: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance)}")
        return x_pca, pca

    def drop_na_columns(data):
        # Drop columns where all values are NaN
        return data.dropna(axis=1, how='all')
    
    def map_diagnosis(diagnosis):
        # Map diagnosis to 0 and 1
        return diagnosis.map({'B': 0, 'M': 1})
    
    def impute_missing_values(data):
        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(data)
        return imputer.transform(data)
    
    def resample_data(x, y):
        # Resample data using SMOTE
        smote = SMOTE()
        return smote.fit_resample(x, y)
    
    def plot_labels_distribution(y):
        # Count the occurrences of each diagnosis
        counts = [sum(y == 0), sum(y == 1)]

        # Plot the distribution of labels using a bar chart
        plt.figure(figsize=(8, 6))
        plt.bar(['Benign', 'Malignant'], counts, color=['blue', 'red'], edgecolor='black')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')
        plt.title('Diagnosis Distribution')
        plt.show()

        return None
    

class EvaluationMetrics:

    def accuracy(y_true, y_pred):
        # Calculate the accuracy of the model
        print('Accuracy Score: ', accuracy_score(y_true, y_pred))
        return None
    
    def confusion_matrix(y_true, y_pred):
        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:\n', cm)
        print('True Positive:', cm[1][1])
        print('False Positive:', cm[0][1])
        print('False Negative:', cm[1][0])
        print('True Negative:', cm[0][0])

        return None
    
    def plot_confusion_matrix(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {title}')
        plt.show()
    
    def classification_report(y_true, y_pred):
        # Generate a classification report
        cr = classification_report(y_true, y_pred=y_pred)
        print('Classification Report:\n', cr)
        return None

    def roc_auc_score(y_true, y_pred):
        # Calculate the AUC-ROC score
        print("ROC AUC Score: ", roc_auc_score(y_true, y_pred), '\n\n')
        return None
    
    def plot_roc_curve(y_true, y_pred, title):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Calculate AUC score
        auc_score = roc_auc_score(y_true, y_pred)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {title}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        return None


def plot_knn_clusters(knn_model, x_train, y_train):
    x_train_pca = PCA(n_components=2).fit_transform(x_train)  # Reduce to 2D
    knn_model.fit(x_train_pca, y_train)
    
    plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolor='k')  # Plot points
    plt.title('KNN Clusters')
    plt.show()



def main() -> None:
    # Load the data
    data = read_csv('./dataset/breast_cancer.csv')
    
    # Drop columns where all values are NaN
    data = PreProcessing.drop_na_columns(data)
    data.drop('id', axis=1, inplace=True)
    
    # Map diagnosis to 0 and 1
    y = PreProcessing.map_diagnosis(data['diagnosis'])
    print(Counter(y))
    PreProcessing.plot_labels_distribution(y)
    
    # Drop diagnosis column
    x = data.drop('diagnosis', axis=1)
    
    # Impute missing values with mean
    x = PreProcessing.impute_missing_values(x)
    
    # Resample data using SMOTE
    x, y = PreProcessing.resample_data(x, y)
    print(Counter(y))
    
    # Plot the distribution of labels
    PreProcessing.plot_labels_distribution(y)

    # Standardize features and get the scaler
    x, scaler = PreProcessing.normalize(x)
    
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    
    # Create and evaluate sklearn KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    y_pred_proba_knn = knn.predict_proba(x_test)[:, 1] # Probability of Malignant

    plot_knn_clusters(knn, x_train, y_train)
    
    print("KNN Classifier Metrics:")
    EvaluationMetrics.accuracy(y_test, y_pred_knn)
    EvaluationMetrics.confusion_matrix(y_test, y_pred_knn)
    EvaluationMetrics.plot_confusion_matrix(y_test, y_pred_knn, 'KNN Classifier')
    EvaluationMetrics.classification_report(y_test, y_pred_knn)
    EvaluationMetrics.roc_auc_score(y_test, y_pred_proba_knn)
    EvaluationMetrics.plot_roc_curve(y_test, y_pred_proba_knn, 'KNN Classifier')

    svc = SVC(kernel='linear', probability=True)
    svc.fit(x_train, y_train)
    y_pred_svc = svc.predict(x_test)
    y_pred_proba_svc = svc.predict_proba(x_test)[:, 1] # Probability of Malignant

    print("SVC Classifier Metrics:")
    EvaluationMetrics.accuracy(y_test, y_pred_svc)
    EvaluationMetrics.confusion_matrix(y_test, y_pred_svc)
    EvaluationMetrics.plot_confusion_matrix(y_test, y_pred_svc, 'SVC Classifier')
    EvaluationMetrics.classification_report(y_test, y_pred_svc)
    EvaluationMetrics.roc_auc_score(y_test, y_pred_proba_svc)
    EvaluationMetrics.plot_roc_curve(y_test, y_pred_proba_svc, 'SVC Classifier')

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_pred_dtc = dtc.predict(x_test)
    y_pred_proba_dtc = dtc.predict_proba(x_test)[:, 1] # Probability of Malignant

    print("Decision Tree Classifier Metrics:")
    EvaluationMetrics.accuracy(y_test, y_pred_dtc)
    EvaluationMetrics.confusion_matrix(y_test, y_pred_dtc)
    EvaluationMetrics.plot_confusion_matrix(y_test, y_pred_dtc, 'Decision Tree Classifier')
    EvaluationMetrics.classification_report(y_test, y_pred_dtc)
    EvaluationMetrics.roc_auc_score(y_test, y_pred_proba_dtc)
    EvaluationMetrics.plot_roc_curve(y_test, y_pred_proba_dtc, 'Decision Tree Classifier')

    # Feature Importance
    feature_names = data.columns[:-1]
    dtc_importance = dtc.feature_importances_
    print("Feature Importance:")
    features = DataFrame({'Feature': feature_names, 'Importance': dtc_importance}, index=range(len(feature_names)))
    features = features.sort_values('Importance', ascending=False)
    print(features)

    # Hyperparameter Tuning
    # ccp_alpha is the complexity parameter used for Minimal Cost-Complexity Pruning and reduces overfitting
    # The best ccp_alpha value is determined using cross-validation's GridSearchCV
    # ccp_alphas = dtc.cost_complexity_pruning_path(x_train, y_train)['ccp_alphas']
    # Criterion is the function to measure the quality of a split
    # Entropy is used to measure the impurity of a node
    dtc2 = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.01)
    dtc2.fit(x_train, y_train)
    y_pred_dtc2 = dtc2.predict(x_test)
    y_pred_proba_dtc2 = dtc2.predict_proba(x_test)[:, 1] # Probability of Malignant

    print("Decision Tree Classifier with Hyperparameter Tuning Metrics:")
    EvaluationMetrics.accuracy(y_test, y_pred_dtc2)
    EvaluationMetrics.confusion_matrix(y_test, y_pred_dtc2)
    EvaluationMetrics.plot_confusion_matrix(y_test, y_pred_dtc2, 'Decision Tree Classifier with Hyperparameter Tuning')
    EvaluationMetrics.classification_report(y_test, y_pred_dtc2)
    EvaluationMetrics.roc_auc_score(y_test, y_pred_proba_dtc2)
    EvaluationMetrics.plot_roc_curve(y_test, y_pred_proba_dtc2, 'Decision Tree Classifier with Hyperparameter Tuning')

    print("Feature Importance for Decision Tree Classifier with Hyperparameter Tuning:")
    features_dtc2 = DataFrame({'Feature': feature_names, 'Importance': dtc2.feature_importances_})
    features_dtc2 = features_dtc2.sort_values('Importance', ascending=False)
    print(features_dtc2)

    # Combine the two DataFrames for side-by-side display
    # combined_features = features.merge(features_dtc2, on='Feature', suffixes=('_dtc', '_dtc2'))

    # print("Feature Importance Comparison:")
    # for index, row in combined_features.iterrows():
    #     print(f"{row['Feature']: <30} - {row['Importance_dtc']: .4f}, {row['Importance_dtc2']: .4f}")

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred_gnb = gnb.predict(x_test)
    y_pred_proba_gnb = gnb.predict_proba(x_test)[:, 1] # Probability of Malignant

    print("Gaussian Naive Bayes Classifier Metrics:")
    EvaluationMetrics.accuracy(y_test, y_pred_gnb)
    EvaluationMetrics.confusion_matrix(y_test, y_pred_gnb)
    EvaluationMetrics.plot_confusion_matrix(y_test, y_pred_gnb, 'Gaussian Naive Bayes Classifier')
    EvaluationMetrics.classification_report(y_test, y_pred_gnb)
    EvaluationMetrics.roc_auc_score(y_test, y_pred_proba_gnb)
    EvaluationMetrics.plot_roc_curve(y_test, y_pred_proba_gnb, 'Gaussian Naive Bayes Classifier')

    param_grid = {
        'var_smoothing' : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    grid_search = GridSearchCV(gnb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    print("Best Estimator:", grid_search.best_estimator_)
    y_pred_gnb2 = grid_search.predict(x_test)
    y_pred_proba_gnb2 = grid_search.predict_proba(x_test)[:, 1] # Probability of Malignant

    print("Gaussian Naive Bayes Classifier with Hyperparameter Tuning Metrics:")
    EvaluationMetrics.accuracy(y_test, y_pred_gnb2)
    EvaluationMetrics.confusion_matrix(y_test, y_pred_gnb2)
    EvaluationMetrics.plot_confusion_matrix(y_test, y_pred_gnb2, 'Gaussian Naive Bayes Classifier with Hyperparameter Tuning')
    EvaluationMetrics.classification_report(y_test, y_pred_gnb2)
    EvaluationMetrics.roc_auc_score(y_test, y_pred_proba_gnb2)
    EvaluationMetrics.plot_roc_curve(y_test, y_pred_proba_gnb2, 'Gaussian Naive Bayes Classifier with Hyperparameter Tuning')

    # Apply PCA
    x_pca, pca = PreProcessing.apply_pca(x, n_components=10) # 10 components explain 95% variance
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca, y, test_size=0.2)
    gnb_pca = GaussianNB()
    gnb_pca.fit(x_train_pca, y_train_pca)
    y_pred_gnb_pca = gnb_pca.predict(x_test_pca)
    y_pred_proba_gnb_pca = gnb_pca.predict_proba(x_test_pca)[:, 1] # Probability of Malignant
    
    print("Gaussian Naive Bayes Classifier with PCA Metrics:")
    EvaluationMetrics.accuracy(y_test_pca, y_pred_gnb_pca)
    EvaluationMetrics.confusion_matrix(y_test_pca, y_pred_gnb_pca)
    EvaluationMetrics.plot_confusion_matrix(y_test_pca, y_pred_gnb_pca, 'Gaussian Naive Bayes Classifier with PCA')
    EvaluationMetrics.classification_report(y_test_pca, y_pred_gnb_pca)
    EvaluationMetrics.roc_auc_score(y_test_pca, y_pred_proba_gnb_pca)
    EvaluationMetrics.plot_roc_curve(y_test_pca, y_pred_proba_gnb_pca, 'Gaussian Naive Bayes Classifier with PCA')


    return None
    

if __name__ == '__main__':
    main()