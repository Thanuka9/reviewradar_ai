-- In psql as postgres on the reviews DB:
ALTER TABLE users
  ADD COLUMN elite           TEXT,
  ADD COLUMN friends         TEXT,
  ADD COLUMN fans            INTEGER,
  ADD COLUMN average_stars   REAL,
  ADD COLUMN compliment_hot      INTEGER,
  ADD COLUMN compliment_more     INTEGER,
  ADD COLUMN compliment_profile  INTEGER,
  ADD COLUMN compliment_cute     INTEGER,
  ADD COLUMN compliment_list     INTEGER,
  ADD COLUMN compliment_note     INTEGER,
  ADD COLUMN compliment_plain    INTEGER,
  ADD COLUMN compliment_cool     INTEGER,
  ADD COLUMN compliment_funny    INTEGER,
  ADD COLUMN compliment_writer   INTEGER,
  ADD COLUMN compliment_photos   INTEGER;
