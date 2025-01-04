# Memory Agent

## Description

This is a simple project that uses a neural network to generate responses to prompts. The neural network is trained on a dataset of conversations, and the responses are generated using the model.

### how enter in the container postgres_db

```bash
docker exec -it postgres_db psql -U example_user -d memory_agent
```

**table creation**

```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL
);
```

**insert values**

```sql
INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'How is my name?', 'Your name is Marco Cobian');
```
