# 🎬 Movie Recommendation System (Python + ML)

A complete, beginner-friendly yet practically applicable **AI-powered Movie Recommendation System** built using:

- **Content-Based Filtering** (Genre Similarity + Cosine Similarity)
- **Collaborative Filtering (SVD)** using Surprise library

This project uses the **MovieLens Dataset** and gives real movie recommendations based on:
✔ Movie genres  
✔ User ratings  
✔ Similarity predictions  

---

## 🚀 Features

### ✅ Content-Based Recommendation
Recommends movies based on **genre similarity** using:
- CountVectorizer
- Cosine Similarity

### ✅ Collaborative Filtering Recommendation
Recommends movies for a user based on:
- Past rated movies
- SVD matrix factorization (Surprise library)

### 🎯 Example Recommendations Output
Movies Loaded: (9742, 3)
Ratings Loaded: (100836, 4)

Building Content-Based Model...

Content Filtering Ready!

Training Collaborative Filtering Model...
Collaborative Model Ready!

Content-based recommendations for: Toy Story (1995)
['Antz (1998)', 'Toy Story 2 (1999)', 'Adventures of Rocky and Bullwinkle, The (2000)', "Emperor's New Groove, The (2000)", 'Monsters, Inc. (2001)']

Collaborative Filtering recommendations for User 1:
['Shawshank Redemption, The (1994)', 'Rear Window (1954)', 'North by Northwest (1959)', 'Wallace & Gromit: The Wrong Trousers (1993)', 'Grand Day Out with Wallace and Gromit, A (1989)']
