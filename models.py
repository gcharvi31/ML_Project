#!/usr/bin/python

from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, \
        RandomForestClassifier, ExtraTreesClassifier, \
        AdaBoostClassifier, VotingClassifier

class EnsemDT():
    
    def __init__(self, subset_features=False,
                    num_base_learners=100,
                    np_ratio=5, feat_dim=100,
                    max_depth=None):
        """ EnsemDT class from paper by Ezzat et al.(2017).
            It is a bagging ensemble of Decision Trees with
            a focus on class imbalance.
            
            :params subset_features: Set true to use feature
                subsetting.
            :params num_base_learners: Number of base learners
                (decision trees) to use.
            :params np_ratio: positive to negative samples
                ratio.
            :params feat_dim: Number of features for 
                subsetting.
            :params max_depth: Maximum depth of each
                individual learner. """
        
        self.num_base_learners = num_base_learners
        self.np_ratio = np_ratio
        self.feat_dim = feat_dim
        self.subset_features = subset_features
        self.max_depth = max_depth
        
        clfs = list()
        for i in range(self.num_base_learners):
            clfs.append(DecisionTreeClassifier(max_depth=self.max_depth))
        self.clfs = clfs
        
        self.clf_fit = False
        
    def fit(self, X_pos, X_neg):
        """ Fit EnsemDT on the dataset.
            
            X_pos and X_neg must be DataFrames
            with label as a column 'y' in each of
            them.
            
            :params X_pos: Dataset with +ve samples.
            :params X_neg: Dataset with -ve samples.
            """
        
        self.X_pos = X_pos
        self.X_neg = X_neg
        
        self.y_pos = X_pos['y']
        self.y_neg = X_neg['y']
        
        self.num_pos = X_pos.shape[0]
        self.num_neg = X_neg.shape[0]
         
        for i in tqdm(range(self.num_base_learners),
                     desc='Training learners...',
                     unit='learners'):
            
            # Random sampling
            X_neg_i = self.X_neg.sample(self.num_pos * self.np_ratio)
            X_pos_i = self.X_pos
            
            # Merge dataset
            X_i = pd.concat([X_neg_i, X_pos_i])
            y_i = X_i['y']
            X_i.drop(['y'], axis=1, inplace=True)
            
            # Feature subsetting
            if self.subset_features:
                X_i = X_i.sample(self.feat_dim, 
                                 axis=1)
            
            self.clfs[i].fit(X_i, y_i)
            
        self.clf_fit = True
            
    def get_scores(self, X_val):
        """ Returns scores of classes. The
            score is directly related to the class
            predicted. 
            
            :params X_val: Validation set (or test). """
        
        if not self.clf_fit:
            raise RuntimeError('Call clf.fit before clf.predict.')
        
        # Create predictions from learners
        preds = list()
        for i in range(self.num_base_learners):
            pred = self.clfs[i].predict(X_val)
            preds.append(pred)
            
        # Average results
        preds = np.vstack(preds)
        preds = preds.T
        
        scores = list()
        for pred in preds:
            scores.append(float(sum(pred))/float(preds.shape[1]))
            
        return scores
    
    def predict(self, X_val):
        """ Predict labels for the given validation
            set (0 or 1). Calls the get_scores function
            for prediction. 
            
            :params X_val: Validation set (or test). """
        
        # Get scores
        preds = list()
        scores = self.get_scores(X_val)

        # Round to predictions
        for score in scores:
            preds.append(round(score))
    
        # Read as numpy array
        preds = np.array(preds).astype('int32')
        
        return preds