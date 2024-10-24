# Movie Recommendation System Using User Similarity

This repository contains a project focused on building a **movie recommendation system** using user similarity techniques based on user ratings. The goal is to create personalized movie suggestions by analyzing user preferences and detecting communities of similar users.

### Project Overview

The recommendation system leverages a **collaborative filtering** approach by computing user-user similarity through techniques such as:
- **Cosine Similarity**
- **Jaccard Index**
- **Pearson Correlation**

By building a **similarity graph** between users, the system is able to recommend movies based on the preferences of similar users, while also exploring the **community structure** of users.

### Data

The dataset used in this project comes from the **<a href='https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset' target='_blank'>Kaggle : Movie Lens Small Latest Dataset</a>**, available on Kaggle. It contains:
- **100,836 ratings** from **610 users** across **9,742 movies**.
- Each user has rated at least 20 movies, and no demographic information is included.

The data spans from **March 29, 1996**, to **September 24, 2018**, and provides both star ratings and free-text tags applied to movies.

### Key Components

1. **Data Preprocessing**: Cleaning and formatting the `ratings.csv` file to ensure it's suitable for similarity analysis.
2. **Similarity Calculation**: Using cosine, Jaccard, and Pearson methods to calculate similarities between users based on their movie ratings.
3. **Graph-Based Analysis**: Building a similarity graph with **Gephi** to visualize relationships between users and detect communities.
4. **Recommendation System**: Generating movie recommendations by leveraging the user similarity scores.
5. **Evaluation**: Assessing the system's performance by measuring the **accuracy** and **relevance** of the recommendations.

### Tools and Libraries
- **Python**: Used for data preprocessing and recommendation system implementation.
- **Gephi**: For visualizing the similarity graph and detecting user communities.
- **Pandas, Numpy, Scikit-learn**: Essential libraries for data manipulation and model implementation.

### Results and Insights

The project successfully demonstrates the power of **similarity-based recommendations** by providing personalized movie suggestions. In addition, the analysis of user relationships via graph structures offers insights into how communities of similar users interact and share preferences.

### Next Steps
- Enhance the model by incorporating other types of recommendation techniques such as **Matrix Factorization**.
- Expand the dataset to test on larger-scale movie databases.
- Improve the recommendation accuracy by integrating more complex evaluation metrics.

### How to Run

1. Clone the repository.
2. Install the required libraries using `requirements.txt`.
3. Run the Jupyter Notebook `recommendation_system.ipynb` to execute the code.

4. ### File Structure

- `data/`: Contains the raw dataset (`ratings.csv`).
- `scripts/`: Python code for building the recommendation system.
- `output/`: Contains the results of the similarity calculations and graph visualizations.
- `requirements.txt`: List of dependencies.
