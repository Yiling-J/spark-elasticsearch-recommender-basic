# spark-elasticsearch-recommender-basic

Basic steps for a simple spark-elasticsearch als recipe recommender

## Installation

apache-spark 2.1.1

elasticsearch 5.5.1

logstash 5.5.1

## Steps

1. Training your model and export factor matrix to database. See: recipe_recommender.py

2. Using logstash to export data to elasticsearch. See: logstash.conf

3. Using elasticsearch api with painless script_score to get results by custom ordering

```
POST http://127.0.0.1:9200/recipes/_search
{
  "query": {
    "function_score": {
      "script_score": {
        "script": {
          "inline": "double total = 0; for (int i = 0; i < doc['vector'].length; ++i) { total += doc['vector'][i] * params.factor[i]; } return total;",
          "params": {
            "factor": <factor from user_factor table>
          }
        }
      }
    }
  }
}
```
