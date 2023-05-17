FROM postgres:14.5
COPY ./sql/* /docker-entrypoint-initdb.d/
CMD ["postgres", "-c", "wal_level=logical"]
