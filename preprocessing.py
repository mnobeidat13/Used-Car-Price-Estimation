import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
import ast
import jenkspy
import bisect
import pickle
import numpy as np

class CarSwitch:
    
    def __init__(self, data):   
        # read the 'model' file which was saved
        with open('finalized_model.sav','rb') as model_file:
            self.reg = pickle.load(model_file)
            self.df = data     
            
            
#     def import_data(self,file_name):
        
#         self.df = pd.read_csv(file_name)
        
    
    def preprocess(self):
        #drop rows with all Nan values 
        self.df = self.df.dropna(how="all")
        
        #Exctracting features
        self.exctract_features()
        
        #drop brands with number of records less than 10
        self.drop_less_than_10()
        
        # Extracting Cars Specifications from Specs column
        self.exctract_specs()
        
        #Creating a weight for each brand using jenks_natural_breaks algorithm
        self.create_weights()
        
        self.final_clean()
        
    
    def final_clean(self):
        self.df = self.df.replace('Bmw', "BMW")
        self.df = self.df.where((self.df["year"] >= 2008) & (self.df["year"] <2019)).dropna()
        #Create dummy variables for vcategorical features
        color_dummies = pd.get_dummies(self.df["Color"])
        specs_dummies = pd.get_dummies(self.df["Specs"])
        brand_dummies = pd.get_dummies(self.df["brand"])
        sub_brand_dummies = pd.get_dummies(self.df["sub-brand"])

        self.df = pd.concat([self.df, specs_dummies, brand_dummies], axis=1)
        
        self.df = self.df.drop(["Color", "Specs", "price", "sub-brand", "brand"], axis=1)

        
        
    def exctract_features(self):
        brands = []
        years = []
        milages = []
        prices = []
        incpections = []
        for brand_name, milage,price, incpection  in zip(self.df["brand-name"], self.df["milage"], self.df["price"], self.df["overall-inc"]):
            years.append(float(brand_name.split()[0]))
            brands.append(str(brand_name.split()[1]))
            milages.append(float(milage.split()[0].replace(",", "")))
            prices.append(float(price.split()[1].replace(",", "")))
            incpections.append(float(incpection.split("/")[0]))
            
        self.df = self.df.drop(["brand-name"], axis=1)
        self.df["brand"] = brands
        self.df["year"] = years
        self.df["milage"] = milages
        self.df["price"] = prices
        self.df["overall-inc"] = incpections
        
    def drop_less_than_10(self):
        #drop brands with number of records less than 10
        temp_df = self.df.groupby("brand").count()["price"]
        brands_to_drop = temp_df.where(temp_df < 10).dropna()
        brands_to_drop = list(brands_to_drop.index)
        #drop brands with number of records less than 10
        self.df = self.df[~self.df["brand"].isin(brands_to_drop)]
        
    def exctract_specs(self):
        list_of_parsed_specs = {"Specs":[], "Color":[], 'Number Of Cylinders':[]}

        specs = pd.read_csv("version2.csv")["Specs"]

        for spec in specs:
            spec = ast.literal_eval(str(spec))

            specs_dicts = []
            for dictionary in spec:
                specs_dicts.append(ast.literal_eval(str(dictionary)))

            spec_list = []
            for dic in specs_dicts:
                for key,value in dic.items():
                    spec_list.append(value)
            spec_dict = {}
            for i, e in enumerate(spec_list):
                if i%2 == 0:
                    spec_dict[e[:-1]] = spec_list[i+1]

            list_of_parsed_specs["Specs"].append(spec_dict["Specs"])
            try:
                list_of_parsed_specs["Color"].append(spec_dict["Color"])
            except:
                list_of_parsed_specs["Color"].append(np.nan)


            try:
                list_of_parsed_specs["Number Of Cylinders"].append(spec_dict["Number Of Cylinders"])
            except:
                list_of_parsed_specs["Number Of Cylinders"].append(np.nan)

        self.df["Specs"] = list_of_parsed_specs["Specs"]
        self.df["no. of cylinders"] = list_of_parsed_specs["Number Of Cylinders"]
        self.df["Color"] = list_of_parsed_specs["Color"]
        self.df = self.df[['brand', 'sub-brand', 'Color', 'year', 'milage', 'Specs', 'no. of cylinders', 'overall-inc',  'price']]
        
    
    def create_weights(self):
        avg_prices = self.df.groupby(["brand"]).mean()["price"]
        prices_df = pd.DataFrame(data={"avg_prices":avg_prices})
        prices_df = prices_df.sort_values("avg_prices")
        
        breaks = jenkspy.jenks_breaks(avg_prices, nb_class=28)
        
        mapping_dict = {}
        for brand, i in zip(prices_df.index, prices_df["avg_prices"]):
            mapping_dict[brand] = bisect.bisect_left(breaks, i)+1

        self.df["brand_weight"] = self.df["brand"].map(mapping_dict)
        mapping_dict
        
        brands_df = {}

        for brand in self.df["brand"]:
            brands_df[brand], _ = [x for _, x in self.df.groupby(self.df['brand'] != brand)]
        
        mapping_dict = {}

        for brand, dataframe in brands_df.items():
            sub_brand_averages = pd.DataFrame(data={"sub_brand_averages":dataframe.groupby(["sub-brand"]).mean()["price"]})
            sub_brand_breaks = jenkspy.jenks_breaks(sub_brand_averages["sub_brand_averages"], nb_class=int(len(sub_brand_averages)-1))
            for i, weight in zip(sub_brand_averages.index, sub_brand_averages["sub_brand_averages"]):
                mapping_dict[i] = bisect.bisect_left(sub_brand_breaks, weight)+1
            self.df["sub_brand_weight"] = self.df["sub-brand"].map(mapping_dict)
            
        self.df = self.df.drop(1095).dropna()
        self.df = self.df.drop(1728).dropna()
        self.df["no. of cylinders"] = self.df["no. of cylinders"].astype(float)
