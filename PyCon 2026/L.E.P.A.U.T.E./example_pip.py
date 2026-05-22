from main import main
from data_access import load_data
import asyncio
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run the pipeline
asyncio.run(main(display_mode="gui,realtime", save_json=True, save_image=True))

# Access data
data = load_data()
for item in data:
    print(f"Lie params: {item['lie_params']}, Loss: {item['loss']}")