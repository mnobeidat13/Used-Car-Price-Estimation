# UsedCarPricePrediction
Data collected from https://carswitch.com/ using https://www.webscraper.io/ chrome extension.
The goal of this project is to predict the prices of used cars sold in UAE based on 8 features;
brand name,	sub-brand name,	Color,	year model,	millage,	Specs(GCC, Japanese, EU, etc)	,number of cylinders and incpection score provided by the website.

Using pandas and sklearn, I was able to clean and prepare data to be fed to a linear regression model, which after trying other machine learning algorithms like SVR and DescisionTree, turned out to be the best.

I could reach 78% training accuracy and 74% test accuracy which is quite low. The reason might be the small amount of data and inaccuracy in data provided by the website.

The part that took most of the time is trying to find an appropriate way to model brand and sub brand variables since they are categotical and have huge impact on the price of a car inspite ofthe other variables. What I did is that i gave each brand a weight based on the average price of the cars in each brand.
