# socket 地址API

字节序问题，现代PC大多采用小端字节序，因此小端字节序又称为主机字节序。

大端字节序也称为网络字节序。JAVA虚拟机采用大端字节序。

## 字节序转化

```c++
#include <netinet/in.h>
//typedef unsigned short int	uint16_t;
//typedef unsigned int		uint32_t;

extern uint32_t ntohl (uint32_t __netlong);
extern uint16_t ntohs (uint16_t __netshort);
extern uint32_t htonl (uint32_t __hostlong);
extern uint16_t htons (uint16_t __hostshort);
```

望文生义：`htonl` 表示host to network long，即将长整型（32位）的主机字节序数据转化为网络字节序数据。

+ 长整型函数通常用来转化IP地址。
+ 短整形函数用来转化端口号。

## 通用socket地址

```c++
#inlcude <bits/socket.h>

//socketaddr.h文件里定义的宏
typedef unsigned short int sa_family_t;
#define	__SOCKADDR_COMMON(sa_prefix) \
  sa_family_t sa_prefix##family
//=====================================


/* Structure describing a generic socket address.  */
struct sockaddr
  {
    __SOCKADDR_COMMON (sa_);	/* Common data: address family and length.  */
    char sa_data[14];		/* Address data.  */
  };

struct sockaddr
  {
    sa_family_t sa_family; //地址族
    char sa_data[14]; //地址，不同地址族有不同含义
  };

//地址族，协议族，sa_data含义
#define AF_UNIX		PF_UNIX    //本地域协议族，文件路径名
#define AF_INET		PF_INET    //TCP/IPv4协议族，16bit端口号 32bit IPv4地址，共6字节
#define AF_INET6	PF_INET6   //TCP/IPv6协议族，26字节，sa_data不够了...
```

## 专用 socket 地址

通用socket地址显然很不好用。。。需要执行繁琐的位操作，Linux为各个协议族提供了专门的 socket 地址结构体。

实际使用的时候需要转化为通用的socket地址类型`sockaddr`（强制转化即可），因为所有socket编程接口使用的地址参数类型都是sockaddr。

```c++
#include <netinet/in.h>

/* Structure describing an Internet socket address.  */
typedef uint16_t in_port_t;
/* Internet address.  */
typedef uint32_t in_addr_t;
struct in_addr
  {
    in_addr_t s_addr;  //IPv4地址，需要网络字节序表示
  };

//IPv4专用socket地址结构
struct sockaddr_in
  {
    __SOCKADDR_COMMON (sin_);    //sa_family_t sa_family; 
    in_port_t sin_port;			/* Port number. . 用网络字节序表示 */
    struct in_addr sin_addr;	 /* Internet address. 见上面 */

    /* Pad to size of `struct sockaddr'.  */
    unsigned char sin_zero[sizeof (struct sockaddr) -
			   __SOCKADDR_COMMON_SIZE -
			   sizeof (in_port_t) -
			   sizeof (struct in_addr)];
  };
```

## IP地址转化（字符串 整数（二进制））

点分十进制字符串<=======>网络字节序整数表示的IPv4地址

```c++
#include <arpa/inet.h>

/* Convert Internet host address from numbers-and-dots notation in CP
   into binary data in network byte order.  */
extern in_addr_t inet_addr (const char *__cp);
/* same as inet_addr, but store the result in the structure INP.  */
extern int inet_aton (const char *__cp, struct in_addr *__inp); //成功返回1 否则0

/* Convert Internet number in IN to ASCII representation.  The return value
   is a pointer to an internal array containing the string.  */
extern char *inet_ntoa (struct in_addr __in);//静态变量存储结果，不可重入
```

使用样例：

```c++
int main()
{
    sockaddr_in addr4;

    uint32_t ip_bin =  inet_addr("192.168.1.1");
    std::cout  << "192.168.1.1 to num: " << ip_bin << std::endl;

    inet_aton("192.168.1.1", &addr4.sin_addr);
    char * ipstr = inet_ntoa(addr4.sin_addr);
    printf("ip str: %s\n", ipstr);
    
    return 0;
}
//控制台输出
// 192.168.1.1 to num: 16885952
// ip str: 192.168.1.1
```

## IP地址转换 更新函数

下面两个转化函数同时适用于IPv4地址和IPv6地址。

```c++
#include <arpa/inet.h>
/* Convert from presentation format of an Internet number in buffer
   starting at CP to the binary network format and store result for
   interface type AF in buffer starting at BUF.  */
// p 互联网号码的显示格式
// n network format
extern int inet_pton (int __af, const char * __cp, void * __buf);

/* Convert a Internet address in binary network format for interface
   type AF in buffer starting at CP to presentation form and place
   result in buffer of length LEN astarting at BUF.  */
extern const char *inet_ntop (int __af, const void * __cp, char *__buf, socklen_t __len);
```

示例：

```c++
//IPv4转化示例
sockaddr_in addr4;
inet_pton(AF_INET, "255.255.255.0", &addr4.sin_addr);
char dst[INET_ADDRSTRLEN ];
const char * client = inet_ntop(AF_INET, &addr4.sin_addr, dst, INET_ADDRSTRLEN);
printf("test num to ip: %s \n", client);
```

# socket API

```c++
#include <sys/socket.h>
```

## 创建socket

```c++
/* Create a new socket of type TYPE in domain DOMAIN, using
   protocol PROTOCOL.  If PROTOCOL is zero, one is chosen automatically.
   Returns a file descriptor for the new socket, or -1 for errors.  */
extern int socket (int __domain, int __type, int __protocol) __THROW;
//参数：协议族 服务类型 通常0
```

## 命名socket

将一个socket与socket地址绑定称为给socket命名。一般服务器要命名socket，因为只有命名后客户端才能知道该如何连接它。客户端则通常不需要命名，而是采用匿名方式，即使用操作系统自动分配的socket地址。

```c++
/* Give the socket FD the local address ADDR (which is LEN bytes long).  */
# define __CONST_SOCKADDR_ARG	const struct sockaddr *
extern int bind (int __fd, __CONST_SOCKADDR_ARG __addr, socklen_t __len) __THROW;
```

bind成功返回1，失败返回-1，并设置errno，常见两种errno：

+ EACCES，被绑定端口是受保护的地址。
+ EADDRINUSE，被绑定的地址正在使用中，比如TIME_WAIT状态。

## 监听socket

socket被命名之后，还不能马上接受客户连接，我们需要使用如下系统调用来**创建一个监听队列**以存放待处理的客户连接。

```c++
/* Prepare to accept connections on socket FD.
   N connection requests will be queued before further requests are refused.
   Returns 0 on success, -1 for errors.  */
extern int listen (int __fd, int __n) __THROW;
```

n 表示内核监听队列的最大长度，监听队列的长度如果超过n，则不再受理新的客户连接。自内核版本2.2之后，它只表示处于完全连接状态的socket的上限。典型值为5。处于半连接状态的socket上限则由`/proc/sys/net/ipv4/tcp_max_syn_backlog`内核参数定义。

不过监听队列中完整连接的上限通常比n要大。

## 例子：

代码清单5-3：

```c++
const char* ip = argv[1];
int port = atoi( argv[2] );
int backlog = atoi( argv[3] );

//1 创建
int sock = socket( PF_INET, SOCK_STREAM, 0 );
assert( sock >= 0 );

struct sockaddr_in address;
bzero( &address, sizeof( address ) );
address.sin_family = AF_INET;
inet_pton( AF_INET, ip, &address.sin_addr );
address.sin_port = htons( port );

//2 命名
int ret = bind( sock, ( struct sockaddr* )&address, sizeof( address ) );
assert( ret != -1 );

//3 监听
ret = listen( sock, backlog );
assert( ret != -1 );

//...

close(sock);
```

## 接受连接

下面系统调用从listen监听队列中接收一个连接：

```c++

```

接口：

```c++
/* Await a connection on socket FD.
   When a connection arrives, open a new socket to communicate with it,
   set *ADDR (which is *ADDR_LEN bytes long) to the address of the connecting
   peer and *ADDR_LEN to the address's actual length, and return the
   new socket's descriptor, or -1 for errors.

   This function is a cancellation point and therefore not marked with
   __THROW.  */
typedef unsigned int socklen_t
# define __SOCKADDR_ARG		struct sockaddr *
extern int accept (int __fd, __SOCKADDR_ARG __addr, socklen_t * __addr_len);
//参数：监听socket，获取远端socket地址，远端socket的长度由addrlen参数指出
```

accept只是从监听队列中取出连接，而不论连接处于何种状态（ESTABLISHED和CLOSE_WAIT）。更不关心任何网络状态的变化。

## 发起连接（客户端）

```c++
/* Open a connection on socket FD to peer at ADDR (which LEN bytes long).
   For connectionless socket types, just set the default address to send to
   and the only address from which to accept transmissions.
   Return 0 on success, -1 for errors.  */
# define __CONST_SOCKADDR_ARG	const struct sockaddr *
extern int connect (int __fd, __CONST_SOCKADDR_ARG __addr, socklen_t __len);
```

## 关闭连接

```c++
#inlcude <unistd.h>

int close(int fd);
```

# TCP数据读写

读的本质来说其实不能是读,在实际中, 具体的接收数据不是由这些调用来进行,是由于系统底层自动完成的。read 也好,recv 也好只负责把数据从底层缓冲copy 到我们指定的位置. 

## \*recv 

对文件的读写操作`read` 和 `write` 同样适用于socket，但是socket编程接口提供了几个专门用于socket数据读写的系统调用，他们增加了对数据读写的控制。其中用于TCP流数据读写的系统调用是：

```c++
#include <sys/socket.h>

//从socket FD 读取 N 字节bytes数据到 BUF
//buf和n分别指定缓冲区的位置的大小
//返回读取到数据的长度，出错返回-1并设置errno，
//返回值可能小于我们期望的长度len，因此我们可能要多次调用recv
//返回0意味着通信对方已经关闭连接了
extern ssize_t recv (int __fd, void *__buf, size_t __n, int __flags);
```

对于读而言::   阻塞和非阻塞的区别在于没有数据到达的时候是否立刻返回． 

+ 阻塞读：

  1、如果没有发现数据在网络缓冲中会一直等待

  2、当发现有数据的时候会把数据读到用户指定的缓冲区，但是如果这个时候读到的数据量比较少，比参数中指定的长度要小，read 并不会一直等待下去，而是立刻返回。由用户决定是否再去read

+ 非阻塞情况下：

  1、如果发现没有数据就直接返回，

  2、如果发现有数据那么也是采用有多少读多少的进行处理．所:read 完一次需要判断读到的数据长度再决定是否还需要再次读取。

## errno设置

阻塞与非阻塞recv返回值没有区分，都是 <0：出错，=0：连接关闭，>0接收到数据大小，特别：返回 值 <0时并且(errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)的情况下认为连接是正常的，继续接收。 

当超时时间到达后，recv会返回错误，也就是-1，而此时的错误码是EAGAIN或者EWOULDBLOCK，POSIX.1-2001上允许两个任意一个出现都行，所以建议在判断错误码上两个都写上。 

 还有一种经常在代码中常见的错误码，那就是EINTER，意思是系统在接收的时候因为收到其他中断信号而被迫返回，不算socket故障，应该继续接收。 

```c++
while(1)
{
    cnt = (int)recv(m_socket, pBuf,RECVSIZE, 0);
    if( cnt >0 )
    {
        //正常处理数据
    }
    else
   {
         if((cnt<0) &&(errno == EAGAIN||errno == EWOULDBLOCK||errno == EINTR)) 
             //这几种错误码，认为连接是正常的，继续接收
        {
            continue;//继续接收数据
        }
        break;//跳出接收循环
    }

}
```

## send

```c++
//往socket fd上写数据，buf和n指定写缓冲区的地址和发送数据的大小
//返回实际写入数据的长度，失败则返回-1
extern ssize_t send (int __fd, const void *__buf, size_t __n, int __flags);
```

写的本质也不是进行发送操作,而是把用户态的数据copy 到系统底层去,然后再由系统进行发送操作,send，write返回成功，只表示数据已经copy 到底层缓冲,而不表示数据已经发出,更不能表示对方端口已经接收到数据。

对于write(或者send)而言， 

+ 阻塞情况下，write会将数据发送完。(不过可能被中断) 
+ 非阻塞写的情况下，是采用可以写多少就写多少的策略 



使用例子：

```c++
//读
char buf[ BUFFER_SIZE ];
memset( buf, '\0', BUFFER_SIZE );
int ret = recv( sockfd, buf, BUFFER_SIZE-1, 0 );
printf( "get %d bytes of content: %s\n", ret, buf );

//写
ret = send( connfd, users[connfd].write_buf, strlen( users[connfd].write_buf ), 0 ); //'\0'结尾
```



## TCP带外数据

[TCP 带外数据（即紧急模式的发送和接受）](https://blog.csdn.net/liushengxi_root/article/details/82563181)

# 高级IO函数

## fcntl函数

file control，提供了对文件描述符的各种控制操作。

```c++
#include <fcntl.h>
int fcntl(int id, int cmd, ...);
```



## pipe

```c++
#include <unistd.h>
//管道容量大小默认是65536字节，可使用fcntl函数修改
int pipe(int fd[2]); //往fd[1]写，从f[0]读
```

## socketpair

```c++
#include <sys/types.h>
#include <sys/socket.h>
extern int socketpair (int __domain, int __type, int __protocol, int __fds[2]);

//前三个函数与socket系统调用的前三个参数完全相同
//但domain只能是本地协议族AF_UNIX, 因为我们仅能在本地使用这个双向管道
//这对文件描述符，即可读又可写
```

## \*readv和writev函数

readv函数将数据从文件描述符读到分散的内存块中，即分散读。writev函数则将多块分散的内存数据一并写入文件描述符。

例如，将http请求的文件分散读到内存块，然后再集中写入到连接socket中。读写是对与文件描述符来说的。

```c++
#include <sys/uio.h>
//在成功时返回读出/写入fd的字节数
ssize_t readv (int __fd, const struct iovec *__iovec, int __count);
ssize_t writev (int __fd, const struct iovec *__iovec, int __count);

/* Structure for scatter/gather I/O.  */
struct iovec
{
    void *iov_base;	/* Pointer to data.  */
    size_t iov_len;	/* Length of data.  */
};
```

### 例子：http应答写入socket

HTTP应答包含1个状态行、多个头部字段、1个空行、和文档的内容。

一块内存：前三部分，另一块文档内容（通过read或者mmap函数）。

```c++
struct iovec iv[2];
iv[0].iov_base = header_buf;
iv[0].iov_len = strlen(header_buf);
iv[1].ivo_base = file_buf;
iv[1].iov_len = file_stat.st_size;
writev(connfd, iv, 2);
```



## mmap函数和munmap函数

mmap函数用于申请一段内存空间。我们可以将这段内存作为：

+ 进程间通信的共享内存，文件数据是没有用的。
+ 也可以将文件直接映射到其中

```c++
#include <sys/mman.h>
void *mmap (void *addr, 
            size_t len, 
            int prot, 
            int flags, 
            int fd, 
            off_t __offset);
//addr允许用户使用某个特定的地址，如果为NULL，则系统自动分配一个地址。
//len指定内存段的长度，会自动调为4k的整数倍，不能为0一般文件多大length就指定多大
//prot设置权限
		//PROT_READ 映射区比必须要有读权限
		//PROT_WRITE
		//PROT_READ | PROT_WRITE
//flags控制内存段内容修改后程序的行为
		//MAP_SHARED 修改了内存数据会同步到磁盘
		//MAP_PRIVATE 修改了内存数据不会同步到磁盘
//fd是映射文件对应的文件描述符，一般通过open系统调用获得
//open文件指定权限应该大于等于mmap第三个参数prot指定的权限

//offset参数设置从文件的何处开始映射（对于不需要读入整个文件的情况），必须为4k 4096的整数倍

//返回值：
//映射区的首地址-调用成功
//调用失败：MAP_FALED

int munmap(void *addr, size_t len); //释放由mmap创建的这段空间
```

```c++
m_file_address = (char *)mmap(0, m_file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
```

创建匿名映射区，没有血缘关系的进程之间不能使用匿名方式，只能借助磁盘文件创建映射区。

```c++
int len=4096;
//创建匿名内存映射区，不指定文件
void *ptr=mmap(NULL,len,PROT_READ | PROT_WRITE,MAP_SHARED | MAP_ANON,-1,0);
```





# IO复用API

## select系统调用



## poll系统调用



## epoll系列

```c++
#inlcude <sys/epoll.h>

//创建内核事件表，需要一个额外的文件描述符来标识
//size参数现在不起作用，只是给内核一个提示，告诉它事件表需要多大
int epoll_create (int __size);
```

下面的函数用于操作epoll的内核事件表：

```c++

int epoll_ctl (int __epfd, int __op, int __fd, struct epoll_event *__event);
//op参数指定操作类型，有如下三个：
/* Valid opcodes ( "op" parameter ) to issue to epoll_ctl().  */
#define EPOLL_CTL_ADD 1	/* Add a file descriptor to the interface.  */
#define EPOLL_CTL_DEL 2	/* Remove a file descriptor from the interface.  */
#define EPOLL_CTL_MOD 3	/* Change file descriptor epoll_event structure.  */
//fd是要操作的文件描述符，event参数指定事件，它的类型是如下结构体：
typedef union epoll_data
{
  void *ptr;
  int fd;          /*使用最多的*/
  uint32_t u32;
  uint64_t u64;
} epoll_data_t;

struct epoll_event
{
  uint32_t events;	/* Epoll events */
  epoll_data_t data;	/* User data variable */
} __EPOLL_PACKED;


//成功时返回：就绪的文件描述符个数
//如果检测到事件，就将就绪的事件从内核事件表中复制到第二个参数events指向的数组中
//极大的提高了索引就绪文件描述符的效率
//maxevents指定最多监听多少个事件
//timeout为设置的超时时间，-1时永远阻塞，知道有某个事件发生，0立即返回
int epoll_wait (int __epfd, struct epoll_event *__events, int __maxevents, int __timeout);
```



使用例子：

```c++
void addfd( int epollfd, int fd, bool enable_et )
{
    epoll_event event;
    event.data.fd = fd;
    event.events = EPOLLIN;
    if( enable_et )
    {
        event.events |= EPOLLET;  //ET模式，边沿触发
    }
    epoll_ctl( epollfd, EPOLL_CTL_ADD, fd, &event );
    setnonblocking( fd );
}

epoll_event events[ MAX_EVENT_NUMBER ]; //1024
int epollfd = epoll_create( 5 );
assert( epollfd != -1 );
addfd( epollfd, listenfd, true );
while( 1 )
{
    int ret = epoll_wait( epollfd, events, MAX_EVENT_NUMBER, -1 );
    for ( int i = 0; i < number; i++ )
    {
        int sockfd = events[i].data.fd;
        /*处理此sockfd上的事件*/
    }
}
```





