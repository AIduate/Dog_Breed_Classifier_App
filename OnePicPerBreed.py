import os

#insert your own path
rootdir = '/models/data/dog_images/train'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))


# In[37]:


keep = []

for dirpath, dirnames, filenames in os.walk(rootdir):
    for filename in [f for f in filenames][:1]:
        keep.append(os.path.join(dirpath, filename))
        


# In[38]:


for dirpath, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if os.path.join(dirpath, filename) not in keep:
            os.remove(os.path.join(dirpath, filename))


# In[ ]:




