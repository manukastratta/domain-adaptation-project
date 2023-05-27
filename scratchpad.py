mean = [129.33003765, 108.17284158, 96.70170372]
std = [68.41791949, 63.12533799, 61.96260248]

mean = [x/255 for x in mean]
std = [x/255 for x in std]

print(mean)
print(std)