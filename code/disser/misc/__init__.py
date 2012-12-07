import os, time


def mpi_progress(view, jobs, num_of_jobs, sleep=10):
    """
    Wrapper function to keep track of progress of mpi jobs.

    Parameters
    ----------
    view : IPython.parallel.client.view
        This is the "view" of the various nodes
    jobs : The handle of your view.map instance
        This is the data structure that provides information on your jobs
    num_of_jobs : int
        The total number of jobs
    sleep: number
        The number of seconds to sleep between iterations

    Returns
    -------
    None

    """
    while jobs.progress < num_of_jobs:
        finished = 0
        view.spin()
        for job in jobs.metadata:
            if job['outputs_ready']: finished += 1
        print('Completed {0} of {1} jobs. ({2} %)'.format(finished,
              num_of_jobs, 100*finished/num_of_jobs))
        time.sleep(sleep)
    print('Completed {0} of {1} jobs. ({2} %)'.format(num_of_jobs,
          num_of_jobs, 100))


def is_leap_year(year):
    """
    Simple function to test for leap years.

    Parameters
    ----------
    year : int
        The four digit year

    Returns
    -------
    1 for leap years
    0 for non-leap years
    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def fsize_check(file_path):
    """
    Simple function to check that the file size is sufficiently large
    to indicated that a file is valid.

    Parameters
    ----------
    file_path : string
        The path to the file to be tested

    Returns
    -------
    1 : if greater than 0.75MB
    0 : if less than or equal to 0.75MB

    """
    try:
        fsize = os.path.getsize(file_path) * 1024**-2
        if fsize > 0.75:
            return 1
    except OSError:
        pass
    return 0


# Variables
se2010 = ['100517', '100518', '100519', '100520', '100521',
          '100524', '100525', '100526', '100527', '100528',
          '100531', '100601', '100602', '100603', '100604',
          '100607', '100608', '100609', '100610', '100611',
          '100614', '100615', '100616', '100617', '100618']

se2011 = ['110427',
          '110509', '110510', '110511', '110512', '110513',
          '110516', '110517', '110518', '110519', '110520',
          '110522',
          '110523', '110524', '110525', '110526', '110527',
          '110530', '110531', '110601', '110602', '110603',
          '110606', '110607', '110608', '110609', '110610']

sedates = se2010 + se2011

se2010_arps_members = ['s4cn_arps']
se2010_nmm_members = ['s4cn_nmm', 's4m3_nmm', 's4m4_nmm', 's4m5_nmm']
se2010_arw_members = ['s4cn_arw', 's4m4_arw', 's4m5_arw', 's4m6_arw',
                      's4m7_arw', 's4m8_arw', 's4m9_arw', 's4m10_arw',
                      's4m11_arw', 's4m12_arw']
se2010_members = se2010_arps_members + se2010_nmm_members + se2010_arw_members


se2011_arps_members = ['s4cn_arps']
se2011_nmm_members = ['s4cn_nmm', 's4m2_nmm', 's4m3_nmm',
                      's4m4_nmm', 's4m5_nmm']
se2011_arw_members = ['s4cn_arw', 's4m4_arw', 's4m5_arw',
                      's4m6_arw', 's4m7_arw', 's4m8_arw',
                      's4m9_arw', 's4m10_arw', 's4m11_arw',
                      's4m12_arw', 's4m13_arw', 's4m14_arw',
                      's4m15_arw', 's4m16_arw', 's4m17_arw',
                      's4m18_arw', 's4m19_arw', 's4m20_arw']
se2011_members = se2011_arps_members + se2011_nmm_members + se2011_arw_members



