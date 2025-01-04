-- Your SQL goes here

CREATE TABLE movies(
    movie_id BIGINT,
    title VARCHAR(2048),
    genres VARCHAR(2048),
    CONSTRAINT pk_movies PRIMARY KEY (movie_id)
);

CREATE TABLE ratings(
   user_id BIGINT,
   movie_id BIGINT,
   rating FLOAT,
   timestamp BIGINT,
 CONSTRAINT pk_ratings PRIMARY KEY (user_id, movie_id)
);