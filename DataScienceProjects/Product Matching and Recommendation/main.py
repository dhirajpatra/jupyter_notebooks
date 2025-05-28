{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "00c27798-b6e2-4b55-8a75-2eda4a8f77f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "import uvicorn\n",
    "\n",
    "app = FastAPI()\n",
    "\n",
    "@app.get(\"/similar/{product_id}\")\n",
    "async def get_similar(product_id: str, n: int = 3):\n",
    "    return {\"similar_products\": find_similar_products(product_id, n)}\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    uvicorn.run(app, host=\"0.0.0.0\", port=8000)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
