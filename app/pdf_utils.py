import aiohttp
from io import BytesIO
from pypdf import PdfReader

async def extract_text_from_pdf_url(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download PDF: HTTP {resp.status}")
            pdf_bytes = await resp.read()
    pdf_file = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    return "\n".join(pages)
