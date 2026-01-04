---
description: Deploy SentiAI to Vercel
---

1. Ensure your code is pushed to a GitHub repository.
2. Log in to [Vercel](https://vercel.com).
3. Click **Add New** -> **Project**.
4. Import your GitHub repository.
5. In the **Environment Variables** section, add:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `SECRET_KEY` (Generate a random one string)
6. Click **Deploy**.
7. Once deployed, copy your Vercel URL (e.g., `https://sentiai.vercel.app`).
8. Go to **Supabase Dashboard** -> **Authentication** -> **URL Configuration**.
9. Update the **Site URL** and add the Vercel URL to the **Redirect URLs** list.
