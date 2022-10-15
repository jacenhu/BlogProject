module.exports = {
    '/program/': [
        {
          title: '数据结构',
          collspsable: true,
          children: [
            '/program/dataStructure/dataStructure',
            '/program/dataStructure/AlgorithmAndDataStructure',
          ],
        },
        {
          title: '前端',
          collspsable: true,
          children: [
            '/program/front/vuePress',
          ],
        },
        {
          title: 'JAVA后端',
          collspsable: true,
          children: [
            '/program/end/开发环境配置',
            '/program/end/springBoot实践',
          ],
        },
        {
          title: 'C++',
          collspsable: true,
          children: [
            '/program/cpp/CppPrimer',
            '/program/cpp/dataStructure',
            '/program/cpp/HTTPIP',
            '/program/cpp/Ceph',
            '/program/cpp/CppKnowledge',
            '/program/cpp/TCPSummary',
            '/program/cpp/linuxCommand',
            '/program/cpp/Abstract',
            '/program/cpp/vscodeCpp',
            '/program/cpp/JD',
            '/program/cpp/mempool',
            '/program/cpp/CodingInterviewSummary',
            '/program/cpp/lowCode',
            '/program/cpp/DataLakeWhat',
            '/program/cpp/designPattern',
          ],
        },
        {
          title: '论文',
          collspsable: true,
          children: [
            '/program/paper/gfs',
            '/program/paper/google3',
          ],
        },
        {
          title: '实践',
          collspsable: true,
          children: [
            '/program/other/harmonyPractice',
          ],
        },
        {
          title: '工具',
          collspsable: true,
          children: [
            '/program/tool/save',
          ],
        },
    ],
    '/books/': [
        {
          title: '书籍',
          collspsable: true,          
          children: [
            '/books/read/人间失格',
            '/books/read/断舍离',
            '/books/read/原则',
            '/books/read/AI未来',
            '/books/read/书单',
          ],
        },
    ],
    // fallback
      '/': [
        '',        /* / */
    ]
  }