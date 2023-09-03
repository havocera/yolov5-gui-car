'''
self.threads = []
self.workers = []

for i in range(5):
    thread = QThread()
    worker = Worker()

    self.threads.append(thread)
    self.workers.append(worker)

    worker.moveToThread(thread)
    thread.started.connect(worker.do_work)

for thread in self.threads:
    thread.start()
'''