import asyncio
import math
import time
import uuid

import httpx


URL = "http://localhost:8080/api/predict"

CONCURRENT_USERS = 10
DURATION_SECONDS = 60

PAYLOAD = {
    "ph": 7.0,
    "Hardness": 204.89,
    "Solids": 20791.32,
    "Chloramines": 7.30,
    "Sulfate": 368.51,
    "Conductivity": 564.30,
    "Organic_carbon": 10.37,
    "Trihalomethanes": 86.99,
    "Turbidity": 2.96,
    "model_type": "rf",
}

latencies = []
success_count = 0
error_count = 0
lock = asyncio.Lock()


def percentile(values, percent):
    if not values:
        return 0

    sorted_values = sorted(values)
    index = math.ceil((percent / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, index)]


async def worker(client, end_time):
    global success_count, error_count

    while time.perf_counter() < end_time:
        request_id = f"load-test-{uuid.uuid4()}"

        start = time.perf_counter()

        try:
            response = await client.post(
                URL,
                json=PAYLOAD,
                headers={"X-Request-ID": request_id},
            )

            latency_ms = (time.perf_counter() - start) * 1000

            async with lock:
                latencies.append(latency_ms)

                if response.status_code == 200:
                    success_count += 1
                else:
                    error_count += 1

        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000

            async with lock:
                latencies.append(latency_ms)
                error_count += 1


async def main():
    print("=== WATER QUALITY LOAD TEST ===")
    print(f"URL              : {URL}")
    print(f"Concurrent users : {CONCURRENT_USERS}")
    print(f"Duration         : {DURATION_SECONDS} seconds")
    print()

    start_time = time.perf_counter()
    end_time = start_time + DURATION_SECONDS

    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            asyncio.create_task(worker(client, end_time))
            for _ in range(CONCURRENT_USERS)
        ]

        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    total_requests = success_count + error_count
    requests_per_second = total_requests / total_time if total_time else 0
    error_rate = (
        (error_count / total_requests) * 100
        if total_requests
        else 0
    )

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)

    print("=== RESULT ===")
    print(f"Total requests   : {total_requests}")
    print(f"Successful       : {success_count}")
    print(f"Failed           : {error_count}")
    print(f"Requests/sec     : {requests_per_second:.2f}")
    print(f"P50 latency      : {p50:.2f} ms")
    print(f"P95 latency      : {p95:.2f} ms")
    print(f"Error rate       : {error_rate:.2f}%")
    print(f"Actual duration  : {total_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())