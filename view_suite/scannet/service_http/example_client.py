#!/usr/bin/env python3
"""
Example client for ScanNet HTTP render service (AsyncUnifiedClient).

Demonstrates how to send a render request and save returned images.
"""
import asyncio
import numpy as np

from view_suite.service_http.async_client import AsyncUnifiedClient


async def main():
    # Initialize client
    client = AsyncUnifiedClient(
        base_url="http://localhost:8767",
        timeout=120.0,
    )

    try:
        # Example: Render a single frame from scene0011_00
        scene_id = "scene0011_00"

        # Camera intrinsics (3x3)
        K = np.array([
            [640.0, 0.0, 320.0],
            [0.0, 480.0, 240.0],
            [0.0, 0.0, 1.0],
        ])

        # Camera extrinsics (4x4, identity)
        T = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Prepare request metadata
        meta = {
            "scene_id": scene_id,
            "tasks": [
                {
                    "mode": "cam_param",
                    "intrinsics": K.tolist(),
                    "extrinsics": T.tolist(),
                    "size": [512, 512],
                }
            ],
        }

        # Send request (IMPORTANT: await)
        print(f"Rendering {len(meta['tasks'])} frames from {scene_id}...")
        response_meta, images = await client.render(meta=meta)

        # Process response
        print("Received response:")
        print(f"  Scene ID:     {response_meta.get('scene_id')}")
        print(f"  Image count:  {len(images)}")
        print(f"  Meta:         {response_meta}")

        # Save images
        for i, img in enumerate(images):
            output_path = f"output_{i}.png"
            img.save(output_path)
            print(f"  Saved: {output_path} ({img.size})")

    finally:
        # IMPORTANT: close the underlying httpx client
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
