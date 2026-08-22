import os, dotenv, httpx, asyncio, struct
dotenv.load_dotenv()

async def test():
    sample_rate = 16000
    num_samples = 16000
    wav_data = bytearray()
    wav_data.extend(b'RIFF')
    data_size = num_samples * 2
    wav_data.extend(struct.pack('<I', 36 + data_size))
    wav_data.extend(b'WAVEfmt ')
    wav_data.extend(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    wav_data.extend(b'data')
    wav_data.extend(struct.pack('<I', data_size))
    wav_data.extend(b'\x00' * data_size)

    key = os.environ['SARVAM_API_KEY']
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            'https://api.sarvam.ai/speech-to-text',
            files={'file': ('test.wav', bytes(wav_data), 'audio/wav')},
            data={'model': 'saaras:v4', 'with_timestamps': 'false'},
            headers={'api-subscription-key': key},
            timeout=30
        )
        print('STATUS:', resp.status_code)
        print('BODY:', resp.text)

asyncio.run(test())
