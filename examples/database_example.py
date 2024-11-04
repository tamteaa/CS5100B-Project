from src.storage.database import DatabaseManager
import json


if __name__ == '__main__':
    # Initialize the DatabaseManager, creating a new database file
    db_manager = DatabaseManager(db_name="agent_data.db", reset_db=True)

    # Insert episode history into the 'episodes' table without manual JSON serialization
    db_manager['episodes'].insert(
        episode_id=1,
        agent_id=101,
        history=[{"action": "move", "result": "success"}]
    )

    db_manager['episodes'].insert(
        episode_id=2,
        agent_id=102,
        history=[{"action": "jump", "result": "fail"}]
    )

    # Fetch all data from 'episodes' table directly via Table instance
    all_episodes = db_manager['episodes'].fetch_all()
    print("All Episodes:", all_episodes)

    # Fetch data for a specific agent from 'episodes' table via Table instance
    agent_episodes = db_manager['episodes'].fetch_by_column('agent_id', 101)
    print("Episodes for agent 101:", agent_episodes)

    # Dynamically create a new 'agents' table
    db_manager.create_table('agents', {
        'agent_id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'level': 'INTEGER'
    })

    # Insert records into the 'agents' table
    db_manager['agents'].insert(agent_id=101, name='Agent A', level=1)
    db_manager['agents'].insert(agent_id=102, name='Agent B', level=2)

    # Fetch all data from the 'agents' table
    all_agents = db_manager['agents'].fetch_all()
    print("All Agents:", all_agents)

    # Close the database connection
    db_manager.close()
