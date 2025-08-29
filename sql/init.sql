-- Table to store incoming user events (clicks, views, purchases)
CREATE TABLE IF NOT EXISTS events (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(128) NOT NULL,
  item_id VARCHAR(128) NOT NULL,
  event_type VARCHAR(32) NOT NULL,
  event_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to store model predictions (for audit / feedback)
CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(128),
  item_id VARCHAR(128),
  score FLOAT,
  model_version VARCHAR(64),
  generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
