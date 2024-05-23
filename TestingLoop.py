import asyncio


async def mouth_start_speaking():
    try:
        while True:
            await asyncio.sleep(0.2)  # Simulate some work
            print("Mouth is speaking...")
    except asyncio.CancelledError:
        print("Mouth speaking task was cancelled")


async def user_chatbot_conversation():
    try:
        while True:
            await asyncio.sleep(1)  # Simulate some work
            print("User chatbot is conversing...")
    except asyncio.CancelledError:
        print("User chatbot conversation task was cancelled")


async def main():
    # Create tasks for both mouth_start_speaking and user_chatbot_conversation
    task_mouth = asyncio.create_task(mouth_start_speaking())
    task_chatbot = asyncio.create_task(user_chatbot_conversation())

    # Continuously run both tasks in a loop, catching any cancellations
    while True:
        try:
            await asyncio.gather(task_mouth, task_chatbot)
        except asyncio.CancelledError:
            continue  # Ignore the cancellation and continue running


# Assuming vts is defined somewhere with necessary methods
asyncio.run(main())  # Run the main coroutine