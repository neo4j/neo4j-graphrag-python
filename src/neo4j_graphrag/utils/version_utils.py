import neo4j


def get_version(
    driver: neo4j.Driver, database: str = "neo4j"
) -> tuple[tuple[int, ...], bool]:
    """
    Retrieves the Neo4j database version and checks if it is running on the Aura platform.

    Args:
        driver (neo4j.Driver): Neo4j Python driver instance to execute the query.
        database (str, optional): The name of the Neo4j database to query. Defaults to "neo4j".

    Returns:
        tuple[tuple[int, ...], bool]:
            - A tuple of integers representing the database version (major, minor, patch) or
                (year, month, patch)  for later versions.
            - A boolean indicating whether the database is hosted on the Aura platform.
    """
    records, _, _ = driver.execute_query(
        "CALL dbms.components()",
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    version = records[0]["versions"][0]
    # drop everything after the '-' first
    version_main, *_ = version.split("-")
    # convert each number between '.' into int
    version_tuple = tuple(map(int, version_main.split(".")))
    # if no patch version, consider it's 0
    if len(version_tuple) < 3:
        version_tuple = (*version_tuple, 0)
    return version_tuple, "aura" in version


def is_version_5_23_or_above(version_tuple: tuple[int, ...]) -> bool:
    """
    Determines if the Neo4j database version is 5.23 or above.

    Args:
        version_tuple (tuple[int, ...]): A tuple of integers representing the database version
            (major, minor, patch) or (year, month, patch) for later versions.

    Returns:
        bool: True if the version is 5.23.0 or above, False otherwise.
    """
    return version_tuple >= (5, 23, 0)


def has_vector_index_support(version_tuple: tuple[int, ...], is_aura: bool) -> bool:
    """
    Checks if a Neo4j database supports vector indexing based on its version and platform.
    For Aura databases, the minimum version is 5.18.0, while for non-Aura databases, it is 5.18.1.

    Args:
        version_tuple (neo4j.Driver): A tuple of integers representing the database version (major, minor, patch) or
                (year, month, patch)  for later versions.
        is_aura (bool): A boolean indicating whether the database is hosted on the Aura platform.

    Returns:
        bool: True if the connected Neo4j database version supports vector indexing, False otherwise.
    """
    if is_aura:
        target_version = (5, 18, 0)
    else:
        target_version = (5, 18, 1)

    return version_tuple >= target_version
