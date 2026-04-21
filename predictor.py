from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class Predictor:
    def __init__(self, dataframe, model, target_column="results"):
        self.df = dataframe
        self.model = model
        self.target_column = target_column
        self.scaler = StandardScaler()
        
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.X_train_scaled, self.X_test_scaled = None, None


    def prepare_data(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Données préparées")


    def train(self):
        self.model.fit(self.X_train_scaled, self.y_train)
        print("Modèle", self.model.__class__.__name__, "entraîné")


    def evaluate(self):
        y_pred = self.model.predict(self.X_test_scaled)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy :", accuracy*100, "%")
        print("Rapport détaillé :")
        print(classification_report(self.y_test, y_pred))
